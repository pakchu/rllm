"""Portfolio optimization over the current fixed alpha pool.

Weights are selected on 2024 test only to reduce eval leakage.  2025 and 2026
YTD are report-only validation windows.  A separate all-window diagnostic ranking
is included for research, but it is not the selection protocol.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.long_component_tp_union_scan import _union_mask
from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask
from training.long_regime_interest_gate_validation import build_interest_features


@dataclass(frozen=True)
class PortfolioOptConfig(LongComboScanConfig):
    output: str = "results/portfolio_opt_new_alpha_pool_2026-07-10.json"
    docs_output: str = "docs/portfolio-opt-new-alpha-pool-2026-07-10.md"
    exclude_from: str = "2026-06-02"
    cost_rate: float = 0.0005
    gross_cap: float = 3.0
    random_samples: int = 3500
    seed: int = 11
    unit_leverage: float = 0.5


SPLITS = [
    ("test2024", "2024-01-01", "2025-01-01"),
    ("eval2025", "2025-01-01", "2026-01-01"),
    ("ytd2026", "2026-01-01", "2026-06-02"),
    ("combined", "2024-01-01", "2026-06-02"),
]

LONG_COMPONENTS = {
    "range_bb90": [("rex_576_range_width_pct", "ge", 0.12959816105499766), ("bb_z", "ge", 1.6850824973528202)],
    "funding10_trend70": [("funding_rate", "le", -0.0000167), ("trend_96", "ge", 0.007485218212390219)],
    "premium20_mom90": [("premium_index_change", "le", -0.00023471), ("htf_1d_return_4", "ge", 0.0940403008961932)],
    "compress05_trend80": [("rex_2016_range_width_pct", "le", 0.05074314472814484), ("trend_24", "ge", 0.004797228904277088)],
}

ALPHAS: dict[str, dict[str, Any]] = {
    "long_funding_compression_premium": {"side": "long", "components": ["funding10_trend70", "compress05_trend80", "premium20_mom90"], "hold": 576, "family": "long_squeeze"},
    "long_range_funding_premium": {"side": "long", "components": ["range_bb90", "funding10_trend70", "premium20_mom90"], "hold": 576, "family": "long_squeeze"},
    "long_minimal_funding_premium": {"side": "long", "components": ["funding10_trend70", "premium20_mom90"], "hold": 576, "family": "long_squeeze"},
    "short_premium_kimchi_union": {"side": "short", "kind": "premium_kimchi_union", "hold": 288, "tp": 0.04, "sl": 0.025, "family": "short_premium_kimchi"},
    "short_fx_stress": {"side": "short", "kind": "fx_stress", "hold": 288, "tp": 0.04, "sl": 0.025, "family": "short_fx"},
    "short_premium_panic": {"side": "short", "kind": "premium_panic", "hold": 288, "tp": 0.04, "sl": 0.025, "family": "short_premium"},
}


def _mask_conditions(features: pd.DataFrame, conditions: list[tuple[str, str, float]]) -> np.ndarray:
    active = np.ones(len(features), dtype=bool)
    for col, op, thr in conditions:
        x = features[col].to_numpy(float)
        active &= np.isfinite(x) & ((x <= float(thr)) if op == "le" else (x >= float(thr)))
    return active


def _alpha_active(features: pd.DataFrame, name: str) -> np.ndarray:
    spec = ALPHAS[name]
    if "components" in spec:
        return _union_mask(features, spec["components"])
    kind = spec["kind"]
    if kind == "fx_stress":
        return _mask_conditions(features, [("htf_3d_return_1", "le", -0.0325294973), ("usdkrw_zscore", "ge", 1.3870063775)])
    if kind == "premium_panic":
        return _mask_conditions(features, [("htf_3d_range_pos", "le", -0.5114186851), ("premium_index_zscore", "le", -1.47209312)])
    if kind == "premium_kimchi_union":
        prem = _mask_conditions(features, [("htf_3d_range_pos", "le", -0.5114186851), ("premium_index_zscore", "le", -1.47209312)])
        kimchi = _mask_conditions(features, [("htf_3d_return_1", "le", -0.0303196833), ("kimchi_premium_change", "le", -0.0046123752)])
        return prem | kimchi
    raise KeyError(name)


def _event_path(
    market: pd.DataFrame,
    signal_pos: int,
    *,
    side: str,
    hold: int,
    cost_rate: float,
    tp: float | None = None,
    sl: float | None = None,
    entry_delay: int = 1,
    leverage: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    n = len(market)
    entry = int(signal_pos) + int(entry_delay)
    max_exit = entry + int(hold)
    if entry >= n - 1 or max_exit >= n:
        return None
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    entry_open = float(opens[entry])
    if entry_open <= 0:
        return None
    ret = np.zeros(n, dtype=np.float64)
    adv = np.zeros(n, dtype=np.float64)
    lev = float(leverage)
    ret[entry] -= float(cost_rate) * lev
    exit_pos = max_exit
    side_mult = 1.0 if side == "long" else -1.0
    realized_reason = "time"
    for j in range(entry, max_exit):
        open_j = float(opens[j])
        if open_j <= 0:
            continue
        if side == "long":
            if sl is not None and float(lows[j]) <= entry_open * (1.0 - sl):
                realized = (entry_open * (1.0 - sl) - open_j) / open_j
                ret[j] += lev * realized
                adv[j] += min(0.0, lev * realized)
                exit_pos = j
                realized_reason = "sl"
                break
            if tp is not None and float(highs[j]) >= entry_open * (1.0 + tp):
                adverse = (float(lows[j]) - open_j) / open_j
                realized = (entry_open * (1.0 + tp) - open_j) / open_j
                adv[j] += min(0.0, lev * adverse)
                ret[j] += lev * realized
                exit_pos = j
                realized_reason = "tp"
                break
            adverse = (float(lows[j]) - open_j) / open_j
            close_ret = (float(opens[j + 1]) - open_j) / open_j
        else:
            if sl is not None and float(highs[j]) >= entry_open * (1.0 + sl):
                realized = (open_j - entry_open * (1.0 + sl)) / open_j
                ret[j] += lev * realized
                adv[j] += min(0.0, lev * realized)
                exit_pos = j
                realized_reason = "sl"
                break
            if tp is not None and float(lows[j]) <= entry_open * (1.0 - tp):
                adverse = (open_j - float(highs[j])) / open_j
                realized = (open_j - entry_open * (1.0 - tp)) / open_j
                adv[j] += min(0.0, lev * adverse)
                ret[j] += lev * realized
                exit_pos = j
                realized_reason = "tp"
                break
            adverse = (open_j - float(highs[j])) / open_j
            close_ret = (open_j - float(opens[j + 1])) / open_j
        adv[j] += min(0.0, lev * adverse)
        ret[j] += lev * close_ret
    ret[exit_pos] -= float(cost_rate) * lev
    realized_total = float(np.prod(1.0 + ret[np.abs(ret) > 0]) - 1.0) if np.any(np.abs(ret) > 0) else 0.0
    return ret, adv, realized_total


def _build_events(market: pd.DataFrame, features: pd.DataFrame, cfg: PortfolioOptConfig) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    dates = pd.to_datetime(market["date"])
    for name, spec in ALPHAS.items():
        active = _alpha_active(features, name)
        hold = int(spec["hold"])
        positions = np.arange(max(143, int(cfg.window_size) - 1), max(0, len(market) - hold - int(cfg.entry_delay_bars) - 1), 12, dtype=np.int64)
        for split, start, end in SPLITS[:3]:
            mask = _split_mask(dates, start, end)
            next_allowed = 0
            for pos in positions[active[positions] & mask[positions]]:
                if int(pos) < next_allowed:
                    continue
                ep = _event_path(
                    market,
                    int(pos),
                    side=spec["side"],
                    hold=hold,
                    cost_rate=float(cfg.cost_rate),
                    tp=spec.get("tp"),
                    sl=spec.get("sl"),
                    entry_delay=int(cfg.entry_delay_bars),
                    leverage=float(cfg.unit_leverage),
                )
                if ep is None:
                    continue
                ret, adv, realized = ep
                exit_pos = min(len(market) - 1, int(pos) + int(cfg.entry_delay_bars) + hold)
                if not mask[min(exit_pos, len(mask) - 1)]:
                    continue
                events.append({"alpha": name, "split": split, "side": spec["side"], "signal_pos": int(pos), "date": str(dates.iloc[int(pos)]), "ret": ret, "adv": adv, "realized_ret": realized})
                next_allowed = exit_pos
    return events


def _arrays(events: list[dict[str, Any]], dates: pd.Series) -> dict[str, Any]:
    names = list(ALPHAS)
    out: dict[str, Any] = {}
    for split, start, end in SPLITS[:3]:
        mask = _split_mask(dates, start, end)
        idx = np.flatnonzero(mask)
        st, en = int(idx[0]), int(idx[-1]) + 1
        mats_r = []
        mats_a = []
        counts = []
        wins = []
        for name in names:
            r = np.zeros(en - st, dtype=np.float64)
            a = np.zeros(en - st, dtype=np.float64)
            c = w = 0
            for e in events:
                if e["split"] == split and e["alpha"] == name:
                    r += e["ret"][st:en]
                    a += e["adv"][st:en]
                    c += 1
                    w += float(e["realized_ret"]) > 0.0
            mats_r.append(r)
            mats_a.append(a)
            counts.append(c)
            wins.append(w)
        R = np.vstack(mats_r)
        A = np.vstack(mats_a)
        active = np.any((R != 0.0) | (A != 0.0), axis=0)
        years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 24 * 3600)
        out[split] = {"R": R[:, active], "A": A[:, active], "counts": np.array(counts), "wins": np.array(wins), "years": years, "active_bars": int(active.sum())}
    return out


def _metric(d: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    names = list(ALPHAS)
    wv = np.array([weights.get(n, 0.0) for n in names], dtype=float)
    r = wv @ d["R"]
    adv = wv @ d["A"]
    if len(r) == 0:
        eq = 1.0; mdd = 0.0
    else:
        eq_path = np.cumprod(np.maximum(0.0, 1.0 + r))
        eq_before = np.r_[1.0, eq_path[:-1]]
        peak_after = np.maximum.accumulate(eq_path)
        peak_before = np.maximum.accumulate(eq_before)
        dd_after = float(np.nanmax(1.0 - eq_path / np.maximum(peak_after, 1e-12)))
        dd_adv = float(np.nanmax(1.0 - (eq_before * np.maximum(0.0, 1.0 + adv)) / np.maximum(peak_before, 1e-12)))
        eq = float(eq_path[-1])
        mdd = max(dd_after, dd_adv) * 100.0
    ret_pct = (eq - 1.0) * 100.0
    cagr = ((eq ** (1.0 / d["years"]) - 1.0) * 100.0) if eq > 0 else -100.0
    trades = int(np.sum(d["counts"][wv != 0]))
    wins = int(np.sum(d["wins"][wv != 0]))
    return {
        "return_pct": ret_pct,
        "cagr_pct": cagr,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": cagr / mdd if mdd > 1e-12 else 0.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "active_bars": d["active_bars"],
        "sleeve_trade_counts": {n: int(c) if weights.get(n, 0.0) > 0 else 0 for n, c in zip(names, d["counts"])}
    }


def _metrics(by: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    return {split: _metric(by[split], weights) for split in ["test2024", "eval2025", "ytd2026"]}


def _score_test_only(stats: dict[str, Any]) -> tuple[Any, ...]:
    t = stats["test2024"]
    # Select on 2024 only; eval/ytd are intentionally not included.
    ok = t["return_pct"] > 0 and t["strict_mdd_pct"] <= 15 and t["cagr_to_strict_mdd"] >= 3 and t["trades"] >= 40
    return (ok, t["cagr_to_strict_mdd"], t["return_pct"], -t["strict_mdd_pct"], t["trades"])


def _score_all_diag(stats: dict[str, Any]) -> tuple[Any, ...]:
    splits = ["test2024", "eval2025", "ytd2026"]
    ok = all(stats[s]["return_pct"] > 0 and stats[s]["strict_mdd_pct"] <= 15 and stats[s]["cagr_to_strict_mdd"] >= 3 for s in splits)
    return (ok, min(stats[s]["cagr_to_strict_mdd"] for s in splits), sum(stats[s]["return_pct"] for s in splits), -max(stats[s]["strict_mdd_pct"] for s in splits))


def _weight_candidates(cfg: PortfolioOptConfig) -> list[dict[str, float]]:
    names = list(ALPHAS)
    out: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()
    def add(w: dict[str, float]) -> None:
        ww = {n: max(0.0, float(w.get(n, 0.0))) for n in names}
        gross = sum(ww.values())
        if gross <= 0 or gross > float(cfg.gross_cap) + 1e-9:
            return
        key = tuple(round(ww[n], 4) for n in names)
        if key not in seen:
            seen.add(key); out.append(ww)
    # Single sleeves and simple family combos.
    for n in names:
        for w in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            add({n: w})
    seeds = [
        {"long_funding_compression_premium": 1.0, "short_premium_kimchi_union": 1.0},
        {"long_funding_compression_premium": 1.0, "short_fx_stress": 1.0},
        {"long_funding_compression_premium": 1.0, "short_premium_kimchi_union": 0.75, "short_fx_stress": 0.75},
        {"long_range_funding_premium": 1.0, "short_premium_kimchi_union": 1.0},
        {"long_minimal_funding_premium": 1.0, "short_premium_kimchi_union": 1.0, "short_fx_stress": 0.5},
    ]
    for s in seeds:
        for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
            add({k: v * scale for k, v in s.items()})
    rng = random.Random(int(cfg.seed))
    for _ in range(int(cfg.random_samples)):
        k = rng.randint(1, min(4, len(names)))
        chosen = rng.sample(names, k)
        raw = np.array([rng.random() ** 1.5 for _ in chosen], dtype=float)
        raw = raw / raw.sum()
        gross = rng.choice([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, float(cfg.gross_cap)])
        add({n: float(v * gross) for n, v in zip(chosen, raw)})
    return out


def _clean_weights(w: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 6) for k, v in w.items() if v > 1e-9}


def run(cfg: PortfolioOptConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    base = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    # Long component scan helper expects module-level COMPONENTS. Monkey patching
    # would be ugly; use local component definitions by assigning into imported module? Instead emulate via conditions.
    import training.long_component_tp_union_scan as lcu
    old_components = dict(lcu.COMPONENTS)
    lcu.COMPONENTS.clear(); lcu.COMPONENTS.update(LONG_COMPONENTS)
    try:
        events = _build_events(market, features, cfg)
    finally:
        lcu.COMPONENTS.clear(); lcu.COMPONENTS.update(old_components)
    dates = pd.to_datetime(market["date"])
    by = _arrays(events, dates)
    rows = []
    for w in _weight_candidates(cfg):
        stats = _metrics(by, w)
        rows.append({"weights": _clean_weights(w), "gross": round(sum(w.values()), 6), "stats": stats, "score_test_only": _score_test_only(stats), "score_all_diag": _score_all_diag(stats)})
    selected = sorted(rows, key=lambda r: r["score_test_only"], reverse=True)
    diagnostic = sorted(rows, key=lambda r: r["score_all_diag"], reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "alphas": ALPHAS,
        "event_counts": {split: dict(Counter(e["alpha"] for e in events if e["split"] == split)) for split in ["test2024", "eval2025", "ytd2026"]},
        "evaluated": len(rows),
        "selection_protocol": "Weights ranked on test2024 only. eval2025/ytd2026 are report-only validation to avoid selecting on eval.",
        "top_selected_test2024": [{k: v for k, v in r.items() if not k.startswith("score_")} for r in selected[:50]],
        "top_all_window_diagnostic": [{k: v for k, v in r.items() if not k.startswith("score_")} for r in diagnostic[:50]],
        "leakage_guard": {"fixed_alpha_thresholds": True, "weight_selection_uses_test2024_only": True, "eval2025_ytd2026_report_only": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _write_doc(cfg, report)
    return report


def _fmt(s: dict[str, Any]) -> str:
    return f"{s['return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trades']}"


def _write_doc(cfg: PortfolioOptConfig, report: dict[str, Any]) -> None:
    lines = [
        "# Portfolio opt with new alpha pool (2026-07-10)", "",
        f"Protocol: {report['selection_protocol']}", "",
        f"Evaluated weight sets: {report['evaluated']:,}; gross cap={cfg.gross_cap}.", "",
        "Metric cell format: `abs_return/CAGR/strict_MDD/CAGR_MDD/trades`.", "",
        "## Top selected by 2024 test only", "",
        "| rank | gross | weights | 2024 test | 2025 eval | 2026 YTD |", "|---:|---:|---|---:|---:|---:|",
    ]
    for i, row in enumerate(report["top_selected_test2024"][:20], 1):
        st = row["stats"]
        lines.append(f"| {i} | {row['gross']:.2f} | `{row['weights']}` | {_fmt(st['test2024'])} | {_fmt(st['eval2025'])} | {_fmt(st['ytd2026'])} |")
    lines += ["", "## All-window diagnostic only", "", "| rank | gross | weights | 2024 test | 2025 eval | 2026 YTD |", "|---:|---:|---|---:|---:|---:|",]
    for i, row in enumerate(report["top_all_window_diagnostic"][:20], 1):
        st = row["stats"]
        lines.append(f"| {i} | {row['gross']:.2f} | `{row['weights']}` | {_fmt(st['test2024'])} | {_fmt(st['eval2025'])} | {_fmt(st['ytd2026'])} |")
    lines += ["", "## Event counts", "", "```json", json.dumps(report["event_counts"], indent=2, ensure_ascii=False), "```", "", "## Interpretation", "", "- If a 2024-selected top row fails 2025/2026, the alpha mix is not robust enough for live sizing despite good in-sample/test selection.", "- All-window diagnostic is useful for research direction only; do not treat it as clean validation."]
    Path(cfg.docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.docs_output).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", default=PortfolioOptConfig.output)
    p.add_argument("--docs-output", default=PortfolioOptConfig.docs_output)
    p.add_argument("--funding-csv", default="")
    p.add_argument("--premium-csv", default="")
    p.add_argument("--exclude-from", default=PortfolioOptConfig.exclude_from)
    p.add_argument("--window-size", type=int, default=PortfolioOptConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=PortfolioOptConfig.entry_delay_bars)
    p.add_argument("--cost-rate", type=float, default=PortfolioOptConfig.cost_rate)
    p.add_argument("--gross-cap", type=float, default=PortfolioOptConfig.gross_cap)
    p.add_argument("--random-samples", type=int, default=PortfolioOptConfig.random_samples)
    p.add_argument("--seed", type=int, default=PortfolioOptConfig.seed)
    p.add_argument("--unit-leverage", type=float, default=PortfolioOptConfig.unit_leverage)
    return p.parse_args()


def main() -> None:
    report = run(PortfolioOptConfig(**vars(parse_args())))
    print(json.dumps({
        "output": report["config"]["output"],
        "docs_output": report["config"]["docs_output"],
        "evaluated": report["evaluated"],
        "event_counts": report["event_counts"],
        "top_selected_test2024": report["top_selected_test2024"][:5],
        "top_all_window_diagnostic": report["top_all_window_diagnostic"][:5],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
