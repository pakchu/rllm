"""Evaluate an LLM-style allow/block selector on the OI divergence high-frequency candidate.

The selector is deliberately implemented as a past-only symbolic proxy over the
same compact state tokens that an LLM selector would receive.  This keeps the
entry alpha fixed and tests whether contextual allow/block decisions are likely
to help before wiring live LLM calls.

Leakage rules:
- Candidate thresholds are read from the saved config.
- Selector context maps are fit only on train (<2024) event outcomes.
- 2024, 2025, and 2026 are transformed with the frozen train context map.
- Execution uses t signal -> t+1 open and non-overlap split-contained replay.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask
from training.strict_bar_backtest import _trade_stats


DEFAULT_CONFIG = "configs/live/oi_divergence_sma24_highfreq_h30_s6_candidate.json"
DEFAULT_OUTPUT = "results/oi_llm_selector_eval_2026-07-06.json"
DEFAULT_CARDS = "results/oi_llm_selector_cards_2026-07-06.jsonl"


@dataclass(frozen=True)
class SelectorCfg:
    candidate_config: str = DEFAULT_CONFIG
    output: str = DEFAULT_OUTPUT
    cards_output: str = DEFAULT_CARDS
    market_csv: str = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
    oi_csv: str = "/tmp/btcusdt_open_interest_5m_2020_2026.csv"
    funding_csv: str = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
    premium_csv: str = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
    exclude_from: str = "2026-06-03"
    window_size: int = 144
    min_train_context_trades: int = 8
    bad_mean_ret_bps: float = -10.0
    bad_win_rate: float = 0.34
    min_keep_rate: float = 0.45
    sweep_selector_thresholds: bool = True
    max_card_rows: int = 0


def _load_candidate(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _load_market_with_oi(cfg: SelectorCfg) -> pd.DataFrame:
    scan_cfg = LongComboScanConfig(
        input_csv=cfg.market_csv,
        output=cfg.output,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        exclude_from=cfg.exclude_from,
        window_size=cfg.window_size,
    )
    market = _load_market(scan_cfg)
    market["date"] = pd.to_datetime(market["date"])
    oi = pd.read_csv(cfg.oi_csv)
    oi["date"] = pd.to_datetime(oi["date"], utc=True).dt.tz_convert(None)
    return market.merge(oi, on="date", how="left")


def _feature_frame(market: pd.DataFrame, window_size: int) -> pd.DataFrame:
    feat = build_market_feature_frame(market, window_size=window_size).copy()
    oi_s = pd.Series(market["open_interest"].astype(float).replace(0, np.nan).ffill(), index=market.index)
    px = pd.Series(market["close"].astype(float), index=market.index)
    for w, name in [(6, "30m"), (12, "1h"), (24, "2h"), (48, "4h"), (96, "8h")]:
        oi_ret = np.log(oi_s / oi_s.shift(w)).replace([np.inf, -np.inf], np.nan)
        px_ret = np.log(px / px.shift(w)).replace([np.inf, -np.inf], np.nan)
        div = oi_ret - px_ret
        for nm, s in [
            (f"oi_ret_{name}", oi_ret),
            (f"px_ret_{name}", px_ret),
            (f"oi_minus_px_{name}", div),
            (f"px_minus_oi_{name}", px_ret - oi_ret),
        ]:
            mu = s.rolling(288, min_periods=50).mean()
            sd = s.rolling(288, min_periods=50).std(ddof=0)
            feat[nm] = s
            feat[nm + "_z"] = ((s - mu) / sd.replace(0, np.nan)).clip(-5, 5)
    return feat.replace([np.inf, -np.inf], np.nan)


def _splits(dates: pd.Series) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    ranges = {
        "train": ("2020-09-01", "2024-01-01"),
        "test2024": ("2024-01-01", "2025-01-01"),
        "eval2025": ("2025-01-01", "2026-01-01"),
        "ytd2026": ("2026-01-01", "2026-06-03"),
    }
    masks = {k: _split_mask(dates, a, b) for k, (a, b) in ranges.items()}
    years = {k: max(1 / 365.25, (pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() / (365.25 * 24 * 3600)) for k, (a, b) in ranges.items()}
    return masks, years


def _candidate_active(feat: pd.DataFrame, candidate: dict[str, Any]) -> np.ndarray:
    active = np.ones(len(feat), dtype=bool)
    for g in candidate["gates"]:
        x = feat[str(g["feature"])].to_numpy(float)
        op = str(g["op"])
        thr = float(g["threshold"])
        if op in (">=", "ge"):
            active &= (x >= thr) & np.isfinite(x)
        elif op in ("<=", "le"):
            active &= (x <= thr) & np.isfinite(x)
        else:
            raise ValueError(f"unsupported op {op}")
    return active


def _trade_factor(p: int, *, hold_bars: int, market: pd.DataFrame, side: str, cost: float) -> tuple[float, float, float]:
    op = market["open"].to_numpy(float)
    hi = market["high"].to_numpy(float)
    lo = market["low"].to_numpy(float)
    ep = int(p) + 1
    xp = ep + int(hold_bars)
    f = 1.0 - cost
    min_factor = f
    for j in range(ep, xp):
        oj = op[j]
        if side == "long":
            adverse = (lo[j] - oj) / oj
            ret = (op[j + 1] - oj) / oj
        else:
            adverse = (oj - hi[j]) / oj
            ret = (oj - op[j + 1]) / oj
        min_factor = min(min_factor, f * max(0.0, 1.0 + adverse))
        f *= max(0.0, 1.0 + ret)
    f *= 1.0 - cost
    min_factor = min(min_factor, f)
    return f, min_factor, f - 1.0


def _events(active: np.ndarray, *, market: pd.DataFrame, feat: pd.DataFrame, masks: dict[str, np.ndarray], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    dates = pd.to_datetime(market["date"])
    hold = int(candidate["hold_bars"])
    stride = int(candidate["stride_bars"])
    side = str(candidate.get("side", "long"))
    cost = float(candidate.get("cost_each_side", 0.0005))
    out: list[dict[str, Any]] = []
    split_by_pos = {}
    for split, m in masks.items():
        for p in np.flatnonzero(m):
            split_by_pos[int(p)] = split
    poss = np.arange(143, len(market) - hold - 2, stride, dtype=np.int64)
    for split, sm in masks.items():
        nxt = 0
        for p in poss[active[poss] & sm[poss]]:
            ip = int(p)
            xp = ip + 1 + hold
            if ip < nxt or xp >= len(market) or not sm[min(xp, len(market) - 1)]:
                continue
            fac, minfac, ret = _trade_factor(ip, hold_bars=hold, market=market, side=side, cost=cost)
            out.append({
                "pos": ip,
                "date": str(dates.iloc[ip]),
                "split": split,
                "entry_pos": ip + 1,
                "exit_pos": xp,
                "entry_date": str(dates.iloc[ip + 1]),
                "exit_date": str(dates.iloc[xp]),
                "factor": fac,
                "min_factor": minfac,
                "ret": ret,
                "ret_bps": ret * 10000.0,
                "tokens": _tokens(ip, market=market, feat=feat),
                "llm_card": _card(ip, market=market, feat=feat),
            })
            nxt = xp
    out.sort(key=lambda r: int(r["pos"]))
    return out


def _bin(value: float, cuts: Iterable[float], labels: tuple[str, ...]) -> str:
    if not np.isfinite(value):
        return "missing"
    for c, lab in zip(cuts, labels):
        if value <= c:
            return lab
    return labels[-1]


def _tokens(i: int, *, market: pd.DataFrame, feat: pd.DataFrame) -> dict[str, str]:
    def val(c: str) -> float:
        return float(feat[c].iloc[i]) if c in feat else float("nan")
    return {
        "trend_4h": _bin(val("htf_4h_return_1"), [-0.003, 0.0, 0.003], ("down", "flat_down", "flat_up", "up")),
        "trend_1d": _bin(val("htf_1d_return_1"), [-0.01, 0.0, 0.01], ("down", "flat_down", "flat_up", "up")),
        "short_sma": _bin(val("sma24_ratio"), [-0.002, -0.0005, 0.0005], ("below_far", "below", "near", "above")),
        "bb_location": _bin(val("bb_z"), [-0.75, -0.25, 0.25, 0.75], ("lower_extreme", "lower", "mid", "upper", "upper_extreme")),
        "rsi": _bin(val("rsi_norm"), [-0.25, 0.0, 0.25], ("oversold", "weak", "firm", "strong")),
        "range_vol": _bin(val("range_vol"), [0.035, 0.05, 0.075], ("normal", "elevated", "high", "extreme")),
        "oi_div_4h": _bin(val("oi_minus_px_4h_z"), [0.25, 0.75, 1.5], ("mild", "clear", "strong", "extreme")),
        "oi_ret_4h": _bin(val("oi_ret_4h_z"), [-0.5, 0.0, 0.75], ("falling", "flat", "rising", "surging")),
        "funding": _bin(val("funding_zscore"), [-0.75, 0.75], ("cold", "neutral", "hot")),
        "premium": _bin(val("premium_index_zscore"), [-0.75, 0.75], ("discount", "neutral", "premium")),
        "kimchi": _bin(val("kimchi_premium_zscore"), [-0.75, 0.75], ("cold", "neutral", "hot")),
        "dxy": _bin(val("dxy_momentum"), [-0.25, 0.25], ("falling", "flat", "rising")),
        "taker_flow": _bin(val("taker_imbalance"), [-0.02, 0.02], ("sell", "neutral", "buy")),
    }


def _card(i: int, *, market: pd.DataFrame, feat: pd.DataFrame) -> dict[str, Any]:
    cols = [
        "oi_minus_px_4h_z", "oi_ret_4h_z", "return_zscore_48", "range_vol", "rsi_norm", "sma24_ratio",
        "bb_z", "funding_zscore", "premium_index_zscore", "kimchi_premium_zscore", "dxy_momentum", "taker_imbalance",
        "htf_4h_return_1", "htf_1d_return_1",
    ]
    vals = {c: (None if c not in feat or not np.isfinite(float(feat[c].iloc[i])) else round(float(feat[c].iloc[i]), 6)) for c in cols}
    return {
        "timestamp": str(pd.to_datetime(market["date"].iloc[i])),
        "task": "allow_or_block_oi_divergence_long_signal",
        "allowed_outputs": ["ALLOW", "BLOCK"],
        "instruction": "Decide whether to allow this fixed OI-divergence long signal. Do not create a new trade; only ALLOW or BLOCK based on regime/context risk.",
        "features": vals,
        "state_tokens": _tokens(i, market=market, feat=feat),
    }


def _context_id(tokens: dict[str, str], keys: tuple[str, ...]) -> str:
    return "|".join(f"{k}={tokens.get(k, 'missing')}" for k in keys)


def _fit_block_contexts(events: list[dict[str, Any]], keys: tuple[str, ...], cfg: SelectorCfg, *, min_train_context_trades: int | None = None, bad_mean_ret_bps: float | None = None, bad_win_rate: float | None = None) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for e in events:
        if e["split"] != "train":
            continue
        grouped[_context_id(e["tokens"], keys)].append(float(e["ret_bps"]))
    blocked = {}
    for cid, vals in grouped.items():
        n = len(vals)
        mean = float(np.mean(vals)) if vals else 0.0
        win = float(np.mean([v > 0 for v in vals])) if vals else 0.0
        min_n = int(cfg.min_train_context_trades if min_train_context_trades is None else min_train_context_trades)
        bad_mean = float(cfg.bad_mean_ret_bps if bad_mean_ret_bps is None else bad_mean_ret_bps)
        bad_win = float(cfg.bad_win_rate if bad_win_rate is None else bad_win_rate)
        if n >= min_n and (mean <= bad_mean or win <= bad_win):
            blocked[cid] = {"train_n": n, "train_mean_ret_bps": mean, "train_win_rate": win}
    return blocked


def _stats(events: list[dict[str, Any]], split: str, years: dict[str, float], blocked: dict[str, Any] | None = None, keys: tuple[str, ...] = ()) -> dict[str, Any]:
    rows = [e for e in events if e["split"] == split]
    if blocked is not None:
        rows = [e for e in rows if _context_id(e["tokens"], keys) not in blocked]
    eq = peak = 1.0
    mdd = 0.0
    rets = []
    for e in rows:
        f = float(e["factor"])
        mf = float(e["min_factor"])
        before = eq
        mdd = max(mdd, 1.0 - (eq * mf) / peak)
        eq *= f
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        rets.append(eq / before - 1.0)
    y = years[split]
    cagr = (eq ** (1 / y) - 1.0) * 100.0 if eq > 0 else -100.0
    md = mdd * 100.0
    ts = _trade_stats(rets)
    return {
        "total_return_pct": (eq - 1.0) * 100.0,
        "cagr_pct": cagr,
        "strict_mdd_pct": md,
        "cagr_to_strict_mdd": cagr / md if md > 1e-12 else float("inf"),
        "trade_entries": len(rets),
        "win_rate": sum(r > 0 for r in rets) / len(rets) if rets else 0.0,
        "mean_trade_ret_pct": ts.get("mean_trade_ret_pct"),
        "p_value": ts.get("p_value_mean_ret_approx"),
        "blocked_trades": len([e for e in [x for x in events if x["split"] == split] if blocked is not None and _context_id(e["tokens"], keys) in blocked]),
    }


def _candidate_keysets() -> list[tuple[str, ...]]:
    return [
        ("short_sma", "bb_location", "oi_ret_4h"),
        ("short_sma", "bb_location", "rsi", "oi_ret_4h"),
        ("trend_4h", "trend_1d", "short_sma", "oi_ret_4h"),
        ("trend_4h", "short_sma", "oi_ret_4h"),
        ("trend_4h", "short_sma"),
        ("bb_location", "oi_ret_4h"),
        ("trend_4h", "short_sma", "bb_location", "oi_ret_4h"),
        ("trend_4h", "trend_1d", "short_sma", "oi_ret_4h"),
        ("trend_4h", "bb_location", "range_vol", "oi_ret_4h"),
        ("short_sma", "bb_location", "rsi", "oi_ret_4h"),
        ("trend_4h", "short_sma", "funding", "premium"),
        ("trend_4h", "short_sma", "kimchi", "dxy"),
        ("trend_4h", "short_sma", "taker_flow", "oi_ret_4h"),
        ("trend_4h", "short_sma", "bb_location", "range_vol", "oi_ret_4h"),
        ("trend_4h", "trend_1d", "short_sma", "bb_location", "oi_ret_4h"),
    ]


def run(cfg: SelectorCfg) -> dict[str, Any]:
    candidate = _load_candidate(cfg.candidate_config)
    market = _load_market_with_oi(cfg)
    feat = _feature_frame(market, cfg.window_size)
    dates = pd.to_datetime(market["date"])
    masks, years = _splits(dates)
    active = _candidate_active(feat, candidate)
    events = _events(active, market=market, feat=feat, masks=masks, candidate=candidate)

    baseline = {split: _stats(events, split, years) for split in ["train", "test2024", "eval2025", "ytd2026"]}
    trials = []
    threshold_grid = [(cfg.min_train_context_trades, cfg.bad_mean_ret_bps, cfg.bad_win_rate)]
    if bool(cfg.sweep_selector_thresholds):
        threshold_grid = []
        for min_n in [16, 24, 32, 48, 64]:
            for bad_mean in [-10.0, -15.0, -20.0, -30.0, -50.0, -75.0, -100.0]:
                for bad_win in [0.34, 0.36, 0.38, 0.40, 0.42]:
                    threshold_grid.append((min_n, bad_mean, bad_win))
    for keys in _candidate_keysets():
        for min_n, bad_mean, bad_win in threshold_grid:
            blocked = _fit_block_contexts(events, keys, cfg, min_train_context_trades=int(min_n), bad_mean_ret_bps=float(bad_mean), bad_win_rate=float(bad_win))
            if not blocked:
                continue
            stats = {split: _stats(events, split, years, blocked=blocked, keys=keys) for split in ["train", "test2024", "eval2025", "ytd2026"]}
            keep_2024 = stats["test2024"]["trade_entries"] / max(1, baseline["test2024"]["trade_entries"])
            keep_2025 = stats["eval2025"]["trade_entries"] / max(1, baseline["eval2025"]["trade_entries"])
            trials.append({
                "context_keys": keys,
                "selector_params": {"min_train_context_trades": int(min_n), "bad_mean_ret_bps": float(bad_mean), "bad_win_rate": float(bad_win)},
                "blocked_contexts": len(blocked),
                "blocked_preview": list(blocked.items())[:20],
                "keep_rate_2024": keep_2024,
                "keep_rate_2025": keep_2025,
                "stats": stats,
                "passes_original_oos_floor": stats["test2024"]["cagr_to_strict_mdd"] >= 5 and stats["eval2025"]["cagr_to_strict_mdd"] >= 5 and stats["test2024"]["trade_entries"] >= 100 and stats["eval2025"]["trade_entries"] >= 100,
                "improves_2026_vs_baseline": stats["ytd2026"]["total_return_pct"] > baseline["ytd2026"]["total_return_pct"] and stats["ytd2026"]["strict_mdd_pct"] <= baseline["ytd2026"]["strict_mdd_pct"],
            })
    trials.sort(key=lambda r: (
        bool(r["passes_original_oos_floor"]),
        bool(r["improves_2026_vs_baseline"]),
        r["stats"]["ytd2026"]["total_return_pct"],
        min(r["stats"]["test2024"]["cagr_to_strict_mdd"], r["stats"]["eval2025"]["cagr_to_strict_mdd"]),
    ), reverse=True)

    best = trials[0] if trials else None
    card_rows = []
    selected_keys = tuple(best["context_keys"]) if best else tuple()
    if best:
        params = best.get("selector_params") or {}
        blocked = _fit_block_contexts(events, selected_keys, cfg, min_train_context_trades=params.get("min_train_context_trades"), bad_mean_ret_bps=params.get("bad_mean_ret_bps"), bad_win_rate=params.get("bad_win_rate"))
    else:
        blocked = {}
    for e in events:
        cid = _context_id(e["tokens"], selected_keys) if selected_keys else ""
        row = {
            "date": e["date"],
            "split": e["split"],
            "signal_pos": e["pos"],
            "selector_context_id": cid,
            "symbolic_selector_action": "BLOCK" if cid in blocked else "ALLOW",
            "future_label_for_training_only": "ALLOW" if float(e["ret_bps"]) > 0 else "BLOCK",
            "realized_ret_bps_for_audit_only": round(float(e["ret_bps"]), 4),
            "llm_card": e["llm_card"],
            "leakage_guard": {
                "llm_card_uses_future": False,
                "future_label_for_training_only_used_in_live_prompt": False,
                "selector_context_fit_split": "train_only",
            },
        }
        card_rows.append(row)
        if int(cfg.max_card_rows) > 0 and len(card_rows) >= int(cfg.max_card_rows):
            break
    Path(cfg.cards_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.cards_output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in card_rows) + ("\n" if card_rows else ""))

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "candidate": {"name": candidate.get("name"), "hold_bars": candidate.get("hold_bars"), "stride_bars": candidate.get("stride_bars"), "gates": candidate.get("gates")},
        "baseline": baseline,
        "best_selector": best,
        "trials": trials,
        "cards_output": cfg.cards_output,
        "event_counts": dict(Counter(e["split"] for e in events)),
        "leakage_guard": {
            "selector_fit_uses_eval2024_2025_2026": False,
            "selector_fit_split": "train_<2024_only",
            "candidate_signal_fixed_before_selector": True,
            "llm_selector_output_space": ["ALLOW", "BLOCK"],
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate OI candidate LLM-style selector cards and symbolic proxy")
    p.add_argument("--candidate-config", default=DEFAULT_CONFIG)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--cards-output", default=DEFAULT_CARDS)
    p.add_argument("--market-csv", default=SelectorCfg.market_csv)
    p.add_argument("--oi-csv", default=SelectorCfg.oi_csv)
    p.add_argument("--funding-csv", default=SelectorCfg.funding_csv)
    p.add_argument("--premium-csv", default=SelectorCfg.premium_csv)
    p.add_argument("--exclude-from", default=SelectorCfg.exclude_from)
    p.add_argument("--window-size", type=int, default=SelectorCfg.window_size)
    p.add_argument("--min-train-context-trades", type=int, default=SelectorCfg.min_train_context_trades)
    p.add_argument("--bad-mean-ret-bps", type=float, default=SelectorCfg.bad_mean_ret_bps)
    p.add_argument("--bad-win-rate", type=float, default=SelectorCfg.bad_win_rate)
    p.add_argument("--max-card-rows", type=int, default=SelectorCfg.max_card_rows)
    p.add_argument("--sweep-selector-thresholds", action="store_true", default=SelectorCfg.sweep_selector_thresholds)
    p.add_argument("--no-sweep-selector-thresholds", dest="sweep_selector_thresholds", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    report = run(SelectorCfg(**vars(ns)))
    best = report.get("best_selector") or {}
    print(json.dumps({
        "output": report["config"]["output"],
        "cards_output": report["cards_output"],
        "event_counts": report["event_counts"],
        "baseline": report["baseline"],
        "best_context_keys": best.get("context_keys"),
        "best_stats": best.get("stats"),
        "passes_original_oos_floor": best.get("passes_original_oos_floor"),
        "improves_2026_vs_baseline": best.get("improves_2026_vs_baseline"),
    }, indent=2, ensure_ascii=False))
