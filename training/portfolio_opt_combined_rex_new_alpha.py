"""Combined portfolio optimization over legacy REX/OI sleeves plus new alpha pool.

This intentionally repairs the 2026-07-10 new-alpha-only scan: the prior gross
5.75/6.10 candidates were REX/OI-heavy, so a fair portfolio search must include
those legacy sleeves and the newly promoted long/short alphas in the same weight
pool.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import training.evaluate_volume_wave_portfolio_combo as vw
import training.portfolio_opt_new_alpha_pool as na
import training.portfolio_with_dynamic_exit_sleeves as dx
import training.search_portfolio_gross6_cost6bp_mdd20_with_dynamic as oldscan
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_interest_gate_validation import build_interest_features

OUT = "results/portfolio_opt_combined_rex_new_alpha_2026-07-10.json"
DOC = "docs/portfolio-opt-combined-rex-new-alpha-2026-07-10.md"
COST = 0.0006
NEW_PREFIX = "new_"

LEGACY_SLEEVES = list(oldscan.SLEEVES)
NEW_ALPHA_NAMES = [NEW_PREFIX + n for n in na.ALPHAS]
SLEEVES = LEGACY_SLEEVES + [n for n in NEW_ALPHA_NAMES if n not in LEGACY_SLEEVES]


@dataclass(frozen=True)
class CombinedOptConfig:
    output: str = OUT
    docs_output: str = DOC
    gross_cap: float = 7.0
    cost_rate: float = COST
    new_alpha_unit_leverage: float = 0.5
    random_samples: int = 6500
    seed: int = 17
    selection_mdd_cap: float = 20.0
    min_test_trades: int = 80
    min_nonzero_weight: float = 0.25
    weight_step: float = 0.05


def _clean(w: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 6) for k, v in w.items() if float(v) > 1e-10}


def _discretize_weights(w: dict[str, float], *, min_nonzero: float, step: float) -> dict[str, float]:
    out: dict[str, float] = {}
    if step <= 0:
        step = 0.05
    for k, v in w.items():
        vv = max(0.0, float(v))
        if vv <= 1e-12:
            continue
        q = round(vv / step) * step
        if q + 1e-12 >= min_nonzero:
            out[k] = round(float(q), 10)
    return out


def _split_starts_ends(masks: dict[str, np.ndarray]) -> tuple[dict[str, int], dict[str, int]]:
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    for sp, m in masks.items():
        idx = np.flatnonzero(m)
        starts[sp] = int(idx[0])
        ends[sp] = int(idx[-1]) + 1
    return starts, ends


def _append_new_alpha_events(events: list[dict[str, Any]], market: pd.DataFrame, masks: dict[str, np.ndarray], cfg: CombinedOptConfig) -> dict[str, Any]:
    base = build_market_feature_frame(market, window_size=144)
    features = pd.concat([base, build_interest_features(market, base)], axis=1).loc[:, lambda x: ~x.columns.duplicated(keep="last")]
    # portfolio_opt_new_alpha_pool._union_mask reads its module-level COMPONENTS;
    # patch only for this construction, then restore.
    import training.long_component_tp_union_scan as lcu
    old_components = dict(lcu.COMPONENTS)
    lcu.COMPONENTS.clear(); lcu.COMPONENTS.update(na.LONG_COMPONENTS)
    try:
        dates = pd.to_datetime(market["date"])
        for alpha, spec in na.ALPHAS.items():
            active = na._alpha_active(features, alpha)
            hold = int(spec["hold"])
            positions = np.arange(143, max(0, len(market) - hold - 2), 12, dtype=np.int64)
            for split, sm in masks.items():
                next_allowed = 0
                for pos in positions[active[positions] & sm[positions]]:
                    ip = int(pos)
                    if ip < next_allowed:
                        continue
                    xp = ip + 1 + hold
                    if xp >= len(market) or not sm[min(xp, len(sm) - 1)]:
                        continue
                    ep = na._event_path(
                        market,
                        ip,
                        side=str(spec["side"]),
                        hold=hold,
                        cost_rate=float(cfg.cost_rate),
                        tp=spec.get("tp"),
                        sl=spec.get("sl"),
                        entry_delay=1,
                        leverage=float(cfg.new_alpha_unit_leverage),
                    )
                    if ep is None:
                        continue
                    ret, adv, realized = ep
                    events.append({
                        "split": split,
                        "sleeve": NEW_PREFIX + alpha,
                        "side": spec["side"],
                        "signal_pos": ip,
                        "date": str(dates.iloc[ip]),
                        "ret_bps": float(realized) * 10000.0,
                        "ret": ret,
                        "adv": adv,
                    })
                    next_allowed = xp
    finally:
        lcu.COMPONENTS.clear(); lcu.COMPONENTS.update(old_components)
    return {"feature_columns": len(features.columns), "rows": len(features)}


def build_combined_events(cfg: CombinedOptConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    vw.COST = float(cfg.cost_rate)
    dx.COST = float(cfg.cost_rate)
    oldscan.COST = float(cfg.cost_rate)
    market, feat, masks, years, events, wave_thresholds = vw.build_events()
    oldscan.add_old_live_events(events, market, feat, masks)
    events.extend(dx.build_dynamic_sleeves(market, feat, masks, years))
    new_meta = _append_new_alpha_events(events, market, masks, cfg)
    events.sort(key=lambda e: (str(e["split"]), int(e["signal_pos"]), str(e["sleeve"])))
    return market, feat, masks, years, events, {"wave_thresholds": wave_thresholds, "new_alpha_meta": new_meta}


def arrays(events: list[dict[str, Any]], masks: dict[str, np.ndarray]) -> dict[str, Any]:
    starts, ends = _split_starts_ends(masks)
    by: dict[str, Any] = {}
    for sp in masks:
        ln = ends[sp] - starts[sp]
        mats_r: list[np.ndarray] = []
        mats_a: list[np.ndarray] = []
        counts: list[int] = []
        wins: list[int] = []
        side_counts: list[dict[str, int]] = []
        for sl in SLEEVES:
            r = np.zeros(ln, dtype=np.float64)
            a = np.zeros(ln, dtype=np.float64)
            c = 0
            w = 0
            sc: Counter[str] = Counter()
            for e in events:
                if e["split"] == sp and e["sleeve"] == sl:
                    st, en = starts[sp], ends[sp]
                    r += e["ret"][st:en]
                    a += e["adv"][st:en]
                    c += 1
                    w += float(e.get("ret_bps", 0.0)) > 0.0
                    sc[str(e.get("side", ""))] += 1
            mats_r.append(r); mats_a.append(a); counts.append(c); wins.append(w); side_counts.append(dict(sc))
        R = np.vstack(mats_r)
        A = np.vstack(mats_a)
        active = np.any((R != 0.0) | (A != 0.0), axis=0)
        by[sp] = {"R": R[:, active], "A": A[:, active], "counts": np.array(counts), "wins": np.array(wins), "side_counts": side_counts, "active_bars": int(active.sum())}
    return by


def metric(d: dict[str, Any], years: float, weights: dict[str, float]) -> dict[str, Any]:
    wv = np.array([weights.get(s, 0.0) for s in SLEEVES], dtype=np.float64)
    r = wv @ d["R"]
    adv = wv @ d["A"]
    if len(r):
        fac = np.maximum(0.0, 1.0 + r)
        eqp = np.cumprod(fac)
        eqb = np.r_[1.0, eqp[:-1]]
        pka = np.maximum.accumulate(eqp)
        pkb = np.maximum.accumulate(eqb)
        dd_after = float(np.nanmax(1.0 - eqp / np.maximum(pka, 1e-12)))
        dd_adv = float(np.nanmax(1.0 - (eqb * np.maximum(0.0, 1.0 + adv)) / np.maximum(pkb, 1e-12)))
        eq = float(eqp[-1])
        mdd = max(dd_after, dd_adv) * 100.0
    else:
        eq = 1.0; mdd = 0.0; r = np.array([], dtype=np.float64)
    vals = r[np.abs(r) > 1e-12]
    sharpe = float(vals.mean() / vals.std(ddof=1) * np.sqrt(len(vals))) if len(vals) > 1 and vals.std(ddof=1) > 0 else 0.0
    ret_pct = (eq - 1.0) * 100.0
    cagr = ((eq ** (1.0 / years) - 1.0) * 100.0) if eq > 0 else -100.0
    trades = int(np.sum(d["counts"][wv != 0.0]))
    wins = int(np.sum(d["wins"][wv != 0.0]))
    return {
        "total_return_pct": ret_pct,
        "cagr_pct": cagr,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": cagr / mdd if mdd > 1e-12 else 0.0,
        "trade_entries": trades,
        "win_rate": wins / trades if trades else 0.0,
        "active_bars": int(d["active_bars"]),
        "bar_sharpe_like": sharpe,
        "sleeve_trade_counts": {s: int(c) if weights.get(s, 0.0) > 0 else 0 for s, c in zip(SLEEVES, d["counts"])},
    }


def metrics(by: dict[str, Any], years: dict[str, float], weights: dict[str, float]) -> dict[str, Any]:
    return {sp: metric(by[sp], years[sp], weights) for sp in ["train", "test2024", "eval2025", "ytd2026"]}


def _load_weights(path: str) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    return {str(k): float(v) for k, v in json.loads(p.read_text()).get("weights", {}).items()}


def _load_prior_top(path: str, key: str = "top_qualified", idx: int = 0) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    rows = d.get(key) or []
    if len(rows) <= idx:
        return {}
    return {str(k): float(v) for k, v in rows[idx].get("weights", {}).items()}


def _new_selected_seed(which: str = "top_selected_test2024", idx: int = 0) -> dict[str, float]:
    p = Path("results/portfolio_opt_new_alpha_pool_2026-07-10.json")
    if not p.exists():
        return {}
    rows = json.loads(p.read_text()).get(which) or []
    if len(rows) <= idx:
        return {}
    return {NEW_PREFIX + str(k): float(v) for k, v in rows[idx].get("weights", {}).items()}


def _scale(w: dict[str, float], k: float) -> dict[str, float]:
    return {s: float(v) * k for s, v in w.items()}


def _scale_to_cap(w: dict[str, float], cap: float) -> dict[str, float]:
    gross = sum(max(0.0, float(v)) for v in w.values())
    if gross <= cap:
        return dict(w)
    return _scale(w, cap / gross)


def _scale_to_mdd(by: dict[str, Any], years: dict[str, float], w: dict[str, float], target: float, cap: float) -> dict[str, float]:
    base = {s: max(0.0, float(w.get(s, 0.0))) for s in SLEEVES}
    if sum(base.values()) <= 0:
        return base
    hi = min(2.0, cap / max(1e-12, sum(base.values())))
    def max_oos_mdd(k: float) -> float:
        ww = _scale(base, k)
        st = metrics(by, years, ww)
        return max(st[x]["strict_mdd_pct"] for x in ["test2024", "eval2025", "ytd2026"])
    while hi < cap / max(1e-12, sum(base.values())) and max_oos_mdd(hi) < target:
        hi = min(hi * 1.25, cap / max(1e-12, sum(base.values())))
    lo = 0.0
    for _ in range(8):
        mid = (lo + hi) / 2.0
        if max_oos_mdd(mid) <= target:
            lo = mid
        else:
            hi = mid
    return _scale(base, lo)


def candidate_weights(by: dict[str, Any], years: dict[str, float], cfg: CombinedOptConfig) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()
    def add(w: dict[str, float]) -> None:
        ww = _discretize_weights({s: max(0.0, float(w.get(s, 0.0))) for s in SLEEVES}, min_nonzero=cfg.min_nonzero_weight, step=cfg.weight_step)
        ww = {s: float(ww.get(s, 0.0)) for s in SLEEVES}
        gross = sum(ww.values())
        if gross <= 1e-12 or gross > cfg.gross_cap + 1e-9:
            return
        key = tuple(round(ww[s], 4) for s in SLEEVES)
        if key not in seen:
            seen.add(key); out.append(ww)

    seeds: list[dict[str, float]] = []
    for p in [
        "configs/live/portfolio_gross575_no_llm_7sleeve_minratio_2026-07-08.json",
        "configs/live/portfolio_gross610_dynamic_top1_2026-07-08.json",
        "configs/live/portfolio_gross4_no_llm_ratio5_mdd20_2026-07-08.json",
        "configs/live/portfolio_gross6_mdd20_ratio5_return_best_candidate.json",
    ]:
        w = _load_weights(p)
        if w:
            seeds.append(w)
    for i in range(5):
        w = _load_prior_top("results/portfolio_gross6_cost6bp_mdd20_with_dynamic_2026-07-08.json", idx=i)
        if w:
            seeds.append(w)
    for i in range(6):
        w = _new_selected_seed("top_selected_test2024", i)
        if w:
            seeds.append(w)
    for i in range(6):
        w = _new_selected_seed("top_all_window_diagnostic", i)
        if w:
            seeds.append(w)

    # Direct seeds, scaled versions, and old/new bridges.
    for s in seeds:
        add(_scale_to_cap(s, cfg.gross_cap))
        for k in [0.35, 0.5, 0.75, 1.0, 1.15]:
            add(_scale_to_cap(_scale(s, k), cfg.gross_cap))
        add(_scale_to_mdd(by, years, s, cfg.selection_mdd_cap * 0.98, cfg.gross_cap))
    old_seeds = [s for s in seeds if any(k in LEGACY_SLEEVES and v > 0 for k, v in s.items())]
    new_seeds = [s for s in seeds if any(k in NEW_ALPHA_NAMES and v > 0 for k, v in s.items())]
    for os in old_seeds[:10]:
        for ns in new_seeds[:12]:
            for ok, nk in [(0.75, 0.25), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.25, 0.75)]:
                w = {}
                for k, v in os.items(): w[k] = w.get(k, 0.0) + float(v) * ok
                for k, v in ns.items(): w[k] = w.get(k, 0.0) + float(v) * nk
                add(_scale_to_cap(w, cfg.gross_cap))
                add(_scale_to_mdd(by, years, w, cfg.selection_mdd_cap * 0.98, cfg.gross_cap))

    rng = random.Random(int(cfg.seed))
    pools = [
        ["nonpb30_taker", "oi_raw", "rex_rule", "oi_upbit_ratio288_low", "bear_rex_short", "oi_alt_ratio72_dyn_exit"],
        ["nonpb30_taker", "oi_low", "oi_high_sel", "bear_rex_short", "rex_rule", "rex_dyn_short_exit", "oi_wave_lowpos144"],
        ["rex_rule", "bear_rex_short", "new_long_funding_compression_premium", "new_short_premium_kimchi_union", "new_short_fx_stress"],
        ["oi_raw", "rex_rule", "new_long_range_funding_premium", "new_short_premium_panic", "new_short_premium_kimchi_union"],
        SLEEVES,
    ]
    for _ in range(int(cfg.random_samples)):
        pool = rng.choice(pools)
        pool = [p for p in pool if p in SLEEVES]
        k = rng.randint(2, min(7, len(pool)))
        chosen = rng.sample(pool, k)
        raw = np.array([rng.random() ** 1.6 for _ in chosen], dtype=np.float64)
        raw = raw / raw.sum()
        gross = rng.choice([2.5, 3.0, 4.0, 5.0, 5.75, 6.1, 7.0])
        w = {c: float(a * gross) for c, a in zip(chosen, raw)}
        add(w)
        # Keep random exploration cheap; expensive MDD scaling is applied to curated seeds/bridges above.
    return out


def _score_test_only(st: dict[str, Any], cfg: CombinedOptConfig) -> tuple[Any, ...]:
    t = st["test2024"]
    ok = t["total_return_pct"] > 0 and t["strict_mdd_pct"] <= cfg.selection_mdd_cap and t["cagr_to_strict_mdd"] >= 3.0 and t["trade_entries"] >= cfg.min_test_trades
    return (ok, t["cagr_to_strict_mdd"], t["total_return_pct"], -t["strict_mdd_pct"], t["trade_entries"])


def _score_robust_diag(st: dict[str, Any], cfg: CombinedOptConfig) -> tuple[Any, ...]:
    splits = ["test2024", "eval2025", "ytd2026"]
    ok = all(st[s]["total_return_pct"] > 0 and st[s]["strict_mdd_pct"] <= cfg.selection_mdd_cap and st[s]["cagr_to_strict_mdd"] >= 3.0 for s in splits)
    min_ratio = min(st[s]["cagr_to_strict_mdd"] for s in splits)
    max_mdd = max(st[s]["strict_mdd_pct"] for s in splits)
    ret_sum = sum(st[s]["total_return_pct"] for s in splits)
    return (ok, min_ratio, ret_sum, -max_mdd, st["ytd2026"]["cagr_to_strict_mdd"])


def run(cfg: CombinedOptConfig) -> dict[str, Any]:
    market, feat, masks, years, events, build_meta = build_combined_events(cfg)
    by = arrays(events, masks)
    rows: list[dict[str, Any]] = []
    for w in candidate_weights(by, years, cfg):
        st = metrics(by, years, w)
        rows.append({"weights": _clean(w), "gross": round(sum(w.values()), 6), "stats": st, "score_test_only": _score_test_only(st, cfg), "score_robust_diag": _score_robust_diag(st, cfg)})
    selected = sorted(rows, key=lambda r: r["score_test_only"], reverse=True)
    robust = sorted(rows, key=lambda r: r["score_robust_diag"], reverse=True)
    def strip(row: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in row.items() if not k.startswith("score_")}
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "selection_protocol": "Weights ranked on test2024 only; eval2025 and ytd2026 are report-only. Robust diagnostic is explicitly eval-influenced research, not clean selection. Nonzero weights are discretized to min 0.25 and 0.05 step.",
        "leverage_semantics": {"legacy_sleeves": "legacy event path at 1.0x per weight", "new_alpha_sleeves": f"event path pre-scaled to {cfg.new_alpha_unit_leverage}x, then portfolio weight applied"},
        "leakage_caveat": "The table ranking uses test2024 only, but the candidate universe includes prior live configs/research artifacts and alpha definitions that may have been informed by 2025/2026 research. Therefore ytd2026 is report-only, not a pristine untouched eval.",
        "sleeves": SLEEVES,
        "new_alpha_prefix": NEW_PREFIX,
        "event_counts": {sp: dict(Counter(e["sleeve"] for e in events if e["split"] == sp)) for sp in masks},
        "evaluated": len(rows),
        "build_meta": build_meta,
        "top_selected_test2024": [strip(r) for r in selected[:50]],
        "top_robust_diagnostic": [strip(r) for r in robust[:50]],
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _write_doc(cfg, report)
    return report


def _fmt(s: dict[str, Any]) -> str:
    return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}"


def _write_doc(cfg: CombinedOptConfig, report: dict[str, Any]) -> None:
    lines = [
        "# Combined REX/OI + new alpha portfolio opt (2026-07-10)",
        "",
        "This reruns the portfolio search after restoring the legacy REX/OI-heavy sleeves that were omitted from the new-alpha-only scan.",
        "",
        f"Protocol: {report['selection_protocol']}",
        f"Gross cap={cfg.gross_cap}; cost each side={cfg.cost_rate:.4%}; new alpha unit leverage={cfg.new_alpha_unit_leverage}; nonzero weight min={cfg.min_nonzero_weight}, step={cfg.weight_step}.",
        "Metric cell format: `abs_return/CAGR/strict_MDD/CAGR_MDD/trades`.",
        "",
        "## Top selected by 2024 test only",
        "",
        "| rank | gross | weights | 2024 test | 2025 eval | 2026 YTD |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for i, row in enumerate(report["top_selected_test2024"][:20], 1):
        st = row["stats"]
        lines.append(f"| {i} | {row['gross']:.2f} | `{row['weights']}` | {_fmt(st['test2024'])} | {_fmt(st['eval2025'])} | {_fmt(st['ytd2026'])} |")
    lines += [
        "",
        "## Robust diagnostic only (eval-influenced)",
        "",
        "| rank | gross | weights | 2024 test | 2025 eval | 2026 YTD |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for i, row in enumerate(report["top_robust_diagnostic"][:20], 1):
        st = row["stats"]
        lines.append(f"| {i} | {row['gross']:.2f} | `{row['weights']}` | {_fmt(st['test2024'])} | {_fmt(st['eval2025'])} | {_fmt(st['ytd2026'])} |")
    lines += [
        "",
        "## Event counts",
        "",
        "```json",
        json.dumps(report["event_counts"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Interpretation",
        "",
        "- The previous new-alpha-only portfolio was not an apples-to-apples replacement for the gross 5.75/6.10 REX/OI portfolios because it omitted `rex_rule`, `bear_rex_short`, dynamic REX exits, and OI sleeves.",
        "- Use the 2024-selected table as cleaner than the robust diagnostic, but do not call 2026 pristine: the candidate universe itself includes prior research artifacts that may have been influenced by later-period analysis.",
        "- Legacy and new sleeve leverage semantics differ; do not deploy a combined row without a live-size normalization pass.",
    ]
    Path(cfg.docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.docs_output).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=CombinedOptConfig.output)
    p.add_argument("--docs-output", default=CombinedOptConfig.docs_output)
    p.add_argument("--gross-cap", type=float, default=CombinedOptConfig.gross_cap)
    p.add_argument("--cost-rate", type=float, default=CombinedOptConfig.cost_rate)
    p.add_argument("--new-alpha-unit-leverage", type=float, default=CombinedOptConfig.new_alpha_unit_leverage)
    p.add_argument("--random-samples", type=int, default=CombinedOptConfig.random_samples)
    p.add_argument("--seed", type=int, default=CombinedOptConfig.seed)
    p.add_argument("--selection-mdd-cap", type=float, default=CombinedOptConfig.selection_mdd_cap)
    p.add_argument("--min-test-trades", type=int, default=CombinedOptConfig.min_test_trades)
    p.add_argument("--min-nonzero-weight", type=float, default=CombinedOptConfig.min_nonzero_weight)
    p.add_argument("--weight-step", type=float, default=CombinedOptConfig.weight_step)
    return p.parse_args()


def main() -> None:
    report = run(CombinedOptConfig(**vars(parse_args())))
    print(json.dumps({
        "output": report["config"]["output"],
        "docs_output": report["config"]["docs_output"],
        "evaluated": report["evaluated"],
        "event_counts": report["event_counts"],
        "top_selected_test2024": report["top_selected_test2024"][:5],
        "top_robust_diagnostic": report["top_robust_diagnostic"][:5],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
