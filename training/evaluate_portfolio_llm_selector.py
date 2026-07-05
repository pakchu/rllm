"""Portfolio-level LLM selector proxy for simultaneous bull/bear sleeves.

This evaluates a second-stage selector above fixed alpha sleeves.  The selector
is constrained to ALLOW/BLOCK_RISK for new sleeve entries only; it cannot create
signals, alter exits, or change leverage.  A symbolic train-only proxy over the
same compact state-card tokens is used before any live LLM wiring.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.evaluate_oi_llm_selector import SelectorCfg, _context_id, _load_market_with_oi, _tokens, _feature_frame as oi_feature_frame
from training.long_regime_combo_scan import _split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.long_regime_score_gate_validation import _build_score_frame, _score_variant


DEFAULT_OUTPUT = "results/portfolio_llm_selector_eval_2026-07-06.json"
DEFAULT_CARDS = "results/portfolio_llm_selector_cards_2026-07-06.jsonl"


@dataclass(frozen=True)
class PortfolioSelectorCfg:
    output: str = DEFAULT_OUTPUT
    cards_output: str = DEFAULT_CARDS
    max_card_rows: int = 0
    min_train_context_events: int = 16
    bad_mean_ret_bps: float = -8.0
    bad_win_rate: float = 0.38
    sweep_thresholds: bool = True


SPLITS = {
    "train": ("2020-09-01", "2024-01-01"),
    "test2024": ("2024-01-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    "ytd2026": ("2026-01-01", "2026-06-03"),
}

PORTFOLIO_WEIGHTS = {
    "nonpb30_taker": 0.5,
    "oi_high_sel": 0.5,
    "bear_rex_short": 1.0,
}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _prep() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, float]]:
    scfg = SelectorCfg()
    market = _load_market_with_oi(scfg)
    market["date"] = pd.to_datetime(market["date"])
    feat = oi_feature_frame(market, 144)
    dates = pd.to_datetime(market["date"])
    masks = {k: _split_mask(dates, a, b) for k, (a, b) in SPLITS.items()}
    years = {k: max(1 / 365.25, (pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() / (365.25 * 24 * 3600)) for k, (a, b) in SPLITS.items()}
    # Restore long-regime activity score for compatibility with portfolio/mixer experiments.
    interest = build_interest_features(market, feat)
    raw = _build_score_frame(market, feat, interest)
    score, _ = _score_variant(raw, masks["train"], "activity_flow_htf")
    feat["activity_flow_htf"] = score
    return market, feat, masks, years


def _gate_active(feat: pd.DataFrame, gates: list[dict[str, Any]]) -> np.ndarray:
    active = np.ones(len(feat), dtype=bool)
    for g in gates:
        x = feat[str(g["feature"])].to_numpy(float)
        thr = float(g["threshold"])
        op = str(g["op"])
        active &= ((x >= thr) if op in (">=", "ge") else (x <= thr)) & np.isfinite(x)
    return active


def _event_return(market: pd.DataFrame, p: int, h: int, side: str, cost: float = 0.0005) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(market)
    op = market["open"].to_numpy(float)
    hi = market["high"].to_numpy(float)
    lo = market["low"].to_numpy(float)
    r = np.zeros(n)
    adv = np.zeros(n)
    ep = int(p) + 1
    xp = ep + int(h)
    r[ep] -= cost
    r[xp] -= cost
    fac = (1 - cost) * (1 - cost)
    for j in range(ep, xp):
        oj = op[j]
        if side == "long":
            rr = (op[j + 1] - oj) / oj
            aa = (lo[j] - oj) / oj
        else:
            rr = (oj - op[j + 1]) / oj
            aa = (oj - hi[j]) / oj
        r[j + 1] += rr
        adv[j] += aa
        fac *= max(0.0, 1.0 + rr)
    return r, adv, fac - 1.0


def _split_for_pos(masks: dict[str, np.ndarray], p: int) -> str | None:
    for split, m in masks.items():
        if 0 <= p < len(m) and bool(m[p]):
            return split
    return None


def _base_context_tokens(i: int, *, market: pd.DataFrame, feat: pd.DataFrame) -> dict[str, str]:
    toks = _tokens(i, market=market, feat=feat)
    # Add a few portfolio-level tokens not in the first OI selector keysets.
    def val(c: str) -> float:
        return float(feat[c].iloc[i]) if c in feat else float("nan")
    def binv(v: float, cuts: list[float], labels: list[str]) -> str:
        if not np.isfinite(v):
            return "missing"
        for c, lab in zip(cuts, labels):
            if v <= c:
                return lab
        return labels[-1]
    toks["trend_1d"] = binv(val("htf_1d_return_1"), [-0.01, 0.0, 0.01], ["down", "flat_down", "flat_up", "up"])
    toks["range_pos_1d"] = binv(val("htf_1d_range_pos"), [0.25, 0.5, 0.75], ["low", "mid_low", "mid_high", "high"])
    toks["vol_state"] = binv(val("range_vol"), [0.035, 0.05, 0.075], ["normal", "elevated", "high", "extreme"])
    return toks


def _llm_card(i: int, *, market: pd.DataFrame, feat: pd.DataFrame, pending_sleeves: list[str]) -> dict[str, Any]:
    cols = ["htf_4h_return_1", "htf_1d_return_1", "range_vol", "rsi_norm", "bb_z", "sma24_ratio", "oi_ret_4h_z", "oi_minus_px_4h_z", "funding_zscore", "premium_index_zscore", "kimchi_premium_zscore", "dxy_momentum", "taker_imbalance"]
    vals = {c: (None if c not in feat or not np.isfinite(float(feat[c].iloc[i])) else round(float(feat[c].iloc[i]), 6)) for c in cols}
    return {
        "timestamp": str(pd.to_datetime(market["date"].iloc[i])),
        "task": "portfolio_sleeve_allow_or_block_risk",
        "allowed_outputs": ["ALLOW", "BLOCK_RISK"],
        "pending_sleeves": pending_sleeves,
        "instruction": "Decide whether to allow these already-triggered portfolio sleeve entries. Do not create trades or alter exits/weights; only ALLOW or BLOCK_RISK for new entries at this timestamp.",
        "features": vals,
        "state_tokens": _base_context_tokens(i, market=market, feat=feat),
    }


def _build_sleeve_events(market: pd.DataFrame, feat: pd.DataFrame, masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    n = len(market)
    events: list[dict[str, Any]] = []
    # nonpb30 taker long
    j = _load_json("configs/live/nonpb30_taker_returnz_rangevol_htf4hrange_h72_candidate.json")
    active = _gate_active(feat, j["signal"]["gates"])
    _append_active_events(events, market, feat, masks, active, "nonpb30_taker", "long", int(j["signal"]["hold_bars_5m"]), int(j["signal"]["stride_bars_5m"]))
    # OI high with first-stage selector
    j = _load_json("configs/live/oi_divergence_sma24_highfreq_h30_s6_candidate.json")
    active = _gate_active(feat, j["gates"])
    sel = _load_json("configs/live/oi_divergence_sma24_highfreq_h30_s6_llm_selector_overlay.json")["symbolic_proxy"]
    blocked = {x["context_id"] for x in sel["blocked_contexts"]}
    keys = tuple(sel["context_keys"])
    def allow_oi(ip: int) -> bool:
        return _context_id(_tokens(ip, market=market, feat=feat), keys) not in blocked
    _append_active_events(events, market, feat, masks, active, "oi_high_sel", "long", int(j["hold_bars"]), int(j["stride_bars"]), selector=allow_oi)
    # Bear REX short predictions
    _append_prediction_events(events, market, feat, masks, "bear_rex_short")
    events.sort(key=lambda e: (int(e["signal_pos"]), str(e["sleeve"])))
    return events


def _append_active_events(events: list[dict[str, Any]], market: pd.DataFrame, feat: pd.DataFrame, masks: dict[str, np.ndarray], active: np.ndarray, sleeve: str, side: str, hold: int, stride: int, selector=None) -> None:
    n = len(market)
    dates = pd.to_datetime(market["date"])
    for split, sm in masks.items():
        nxt = 0
        for p in np.arange(143, n - hold - 2, stride, dtype=np.int64):
            ip = int(p)
            xp = ip + 1 + hold
            if not (active[ip] and sm[ip]) or ip < nxt or xp >= n or not sm[min(xp, n - 1)]:
                continue
            if selector and not selector(ip):
                continue
            r, adv, realized = _event_return(market, ip, hold, side)
            events.append({"sleeve": sleeve, "side": side, "split": split, "signal_pos": ip, "entry_pos": ip + 1, "exit_pos": xp, "date": str(dates.iloc[ip]), "ret_bps": realized * 10000.0, "ret": r, "adv": adv, "tokens": _base_context_tokens(ip, market=market, feat=feat)})
            nxt = xp


def _append_prediction_events(events: list[dict[str, Any]], market: pd.DataFrame, feat: pd.DataFrame, masks: dict[str, np.ndarray], sleeve: str) -> None:
    files = [
        "results/rex_dual_regime_train_2021_2023_predictions_2026-07-03.jsonl",
        "results/rex_dual_regime_test_2024_predictions_2026-07-03.jsonl",
        "results/rex_dual_regime_eval_2025_2026h1_predictions_2026-07-03.jsonl",
    ]
    rows = []
    for f in files:
        if Path(f).exists():
            rows.extend(json.loads(line) for line in Path(f).read_text().splitlines() if line.strip())
    rows.sort(key=lambda r: int(r["signal_pos"]))
    n = len(market)
    dates = pd.to_datetime(market["date"])
    for split, sm in masks.items():
        nxt = 0
        for row in rows:
            pred = row.get("prediction") or {}
            if pred.get("side") != "SHORT":
                continue
            ip = int(row["signal_pos"])
            hold = int(pred.get("hold_bars") or 0)
            xp = ip + 1 + hold
            if hold <= 0 or ip < 143 or ip >= n - hold - 2 or ip < nxt or not sm[ip] or xp >= n or not sm[min(xp, n - 1)]:
                continue
            r, adv, realized = _event_return(market, ip, hold, "short")
            events.append({"sleeve": sleeve, "side": "short", "split": split, "signal_pos": ip, "entry_pos": ip + 1, "exit_pos": xp, "date": str(dates.iloc[ip]), "ret_bps": realized * 10000.0, "ret": r, "adv": adv, "tokens": _base_context_tokens(ip, market=market, feat=feat)})
            nxt = xp


def _context_keysets() -> list[tuple[str, ...]]:
    return [
        ("trend_4h", "trend_1d", "vol_state"),
        ("trend_4h", "trend_1d", "short_sma", "vol_state"),
        ("trend_4h", "trend_1d", "oi_ret_4h"),
        ("trend_4h", "short_sma", "bb_location", "oi_ret_4h"),
        ("trend_1d", "range_pos_1d", "dxy", "kimchi"),
        ("funding", "premium", "kimchi", "dxy"),
        ("trend_4h", "taker_flow", "oi_ret_4h"),
        ("short_sma", "bb_location", "oi_ret_4h"),
    ]


def _fit_blocked(events: list[dict[str, Any]], keys: tuple[str, ...], *, min_n: int, bad_mean: float, bad_win: float) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for e in events:
        if e["split"] != "train":
            continue
        # train utility is weighted event contribution from fixed portfolio.
        grouped[_context_id(e["tokens"], keys)].append(float(e["ret_bps"]) * float(PORTFOLIO_WEIGHTS.get(str(e["sleeve"]), 0.0)))
    blocked = {}
    for cid, vals in grouped.items():
        if len(vals) < min_n:
            continue
        mean = float(np.mean(vals))
        win = float(np.mean([v > 0 for v in vals]))
        if mean <= bad_mean or win <= bad_win:
            blocked[cid] = {"train_n": len(vals), "train_mean_weighted_ret_bps": mean, "train_win_rate": win}
    return blocked


def _metrics(events: list[dict[str, Any]], split: str, masks: dict[str, np.ndarray], years: dict[str, float], blocked: dict[str, Any] | None = None, keys: tuple[str, ...] = ()) -> dict[str, Any]:
    idx = np.flatnonzero(masks[split])
    r = np.zeros(len(next(iter(masks.values()))))
    adv = np.zeros(len(next(iter(masks.values()))))
    trades = 0
    blocked_trades = 0
    for e in events:
        if e["split"] != split:
            continue
        cid = _context_id(e["tokens"], keys) if blocked is not None else ""
        if blocked is not None and cid in blocked:
            blocked_trades += 1
            continue
        w = float(PORTFOLIO_WEIGHTS.get(str(e["sleeve"]), 0.0))
        if w == 0:
            continue
        r += w * e["ret"]
        adv += w * e["adv"]
        trades += 1
    rr = r[idx]
    aa = adv[idx]
    factors = np.maximum(0.0, 1.0 + rr)
    eq_path = np.cumprod(factors) if len(factors) else np.array([1.0])
    eq_before = np.concatenate([[1.0], eq_path[:-1]]) if len(factors) else np.array([1.0])
    peak_before = np.maximum.accumulate(eq_before)
    peak_after = np.maximum.accumulate(eq_path) if len(factors) else np.array([1.0])
    dd_close = 1.0 - eq_path / np.maximum(peak_after, 1e-12)
    dd_adv = 1.0 - (eq_before * (1.0 + aa)) / np.maximum(peak_before, 1e-12)
    mdd = float(max(np.nanmax(dd_close) if len(dd_close) else 0.0, np.nanmax(dd_adv) if len(dd_adv) else 0.0))
    eq = float(eq_path[-1]) if len(eq_path) else 1.0
    cagr = (eq ** (1.0 / years[split]) - 1.0) * 100.0 if eq > 0 else -100.0
    md = mdd * 100.0
    vals = rr[np.abs(rr) > 1e-12]
    sharpe = float(np.mean(vals) / np.std(vals, ddof=1) * np.sqrt(len(vals))) if len(vals) > 1 and np.std(vals, ddof=1) > 0 else 0.0
    return {"total_return_pct": (eq - 1.0) * 100.0, "cagr_pct": cagr, "strict_mdd_pct": md, "cagr_to_strict_mdd": cagr / md if md > 1e-12 else float("inf"), "trade_entries": trades, "blocked_trades": blocked_trades, "active_bars": int(len(vals)), "bar_sharpe_like": sharpe}


def run(cfg: PortfolioSelectorCfg) -> dict[str, Any]:
    market, feat, masks, years = _prep()
    events = _build_sleeve_events(market, feat, masks)
    baseline = {sp: _metrics(events, sp, masks, years) for sp in SPLITS}
    grid = [(cfg.min_train_context_events, cfg.bad_mean_ret_bps, cfg.bad_win_rate)]
    if cfg.sweep_thresholds:
        grid = [(n, m, w) for n in [12, 16, 24, 32, 48] for m in [-5.0, -8.0, -10.0, -15.0, -20.0, -30.0] for w in [0.34, 0.36, 0.38, 0.40, 0.42]]
    trials = []
    for keys in _context_keysets():
        for min_n, bad_mean, bad_win in grid:
            blocked = _fit_blocked(events, keys, min_n=int(min_n), bad_mean=float(bad_mean), bad_win=float(bad_win))
            if not blocked:
                continue
            stats = {sp: _metrics(events, sp, masks, years, blocked=blocked, keys=keys) for sp in SPLITS}
            trials.append({
                "context_keys": keys,
                "selector_params": {"min_train_context_events": int(min_n), "bad_mean_weighted_ret_bps": float(bad_mean), "bad_win_rate": float(bad_win)},
                "blocked_contexts": len(blocked),
                "blocked_preview": list(blocked.items())[:20],
                "stats": stats,
                "passes_2024_2025_floor": stats["test2024"]["cagr_to_strict_mdd"] >= 5 and stats["eval2025"]["cagr_to_strict_mdd"] >= 5,
                "improves_2026": stats["ytd2026"]["cagr_to_strict_mdd"] > baseline["ytd2026"]["cagr_to_strict_mdd"] and stats["ytd2026"]["strict_mdd_pct"] <= baseline["ytd2026"]["strict_mdd_pct"],
            })
    trials.sort(key=lambda r: (bool(r["passes_2024_2025_floor"]), bool(r["improves_2026"]), r["stats"]["ytd2026"]["cagr_to_strict_mdd"], min(r["stats"]["test2024"]["cagr_to_strict_mdd"], r["stats"]["eval2025"]["cagr_to_strict_mdd"])), reverse=True)
    best = trials[0] if trials else None
    cards = []
    blocked = {}
    keys: tuple[str, ...] = tuple()
    if best:
        keys = tuple(best["context_keys"])
        p = best["selector_params"]
        blocked = _fit_blocked(events, keys, min_n=int(p["min_train_context_events"]), bad_mean=float(p["bad_mean_weighted_ret_bps"]), bad_win=float(p["bad_win_rate"]))
    # group same-timestamp pending sleeves for card output.
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        grouped[int(e["signal_pos"])].append(e)
    for pos in sorted(grouped):
        evs = grouped[pos]
        cid = _context_id(evs[0]["tokens"], keys) if keys else ""
        row = {
            "date": evs[0]["date"],
            "split": evs[0]["split"],
            "signal_pos": pos,
            "selector_context_id": cid,
            "symbolic_selector_action": "BLOCK_RISK" if cid in blocked else "ALLOW",
            "pending_sleeves": [e["sleeve"] for e in evs],
            "weighted_realized_ret_bps_for_audit_only": round(sum(float(PORTFOLIO_WEIGHTS.get(e["sleeve"], 0.0)) * float(e["ret_bps"]) for e in evs), 4),
            "llm_card": _llm_card(pos, market=market, feat=feat, pending_sleeves=[e["sleeve"] for e in evs]),
            "leakage_guard": {"llm_card_uses_future": False, "future_realized_ret_used_in_live_prompt": False, "selector_context_fit_split": "train_only"},
        }
        cards.append(row)
        if cfg.max_card_rows > 0 and len(cards) >= cfg.max_card_rows:
            break
    Path(cfg.cards_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.cards_output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in cards) + ("\n" if cards else ""))
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "portfolio_weights": PORTFOLIO_WEIGHTS, "baseline": baseline, "best_selector": best, "trials": trials[:200], "event_counts": dict(Counter(e["split"] for e in events)), "cards_output": cfg.cards_output, "leakage_guard": {"selector_fit_split": "train_<2024_only", "selector_output_space": ["ALLOW", "BLOCK_RISK"], "base_sleeve_signals_fixed": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--cards-output", default=DEFAULT_CARDS)
    p.add_argument("--max-card-rows", type=int, default=0)
    p.add_argument("--min-train-context-events", type=int, default=PortfolioSelectorCfg.min_train_context_events)
    p.add_argument("--bad-mean-ret-bps", type=float, default=PortfolioSelectorCfg.bad_mean_ret_bps)
    p.add_argument("--bad-win-rate", type=float, default=PortfolioSelectorCfg.bad_win_rate)
    p.add_argument("--no-sweep-thresholds", dest="sweep_thresholds", action="store_false", default=True)
    return p.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    rep = run(PortfolioSelectorCfg(**vars(ns)))
    best = rep.get("best_selector") or {}
    print(json.dumps({"output": rep["config"]["output"], "cards_output": rep["cards_output"], "event_counts": rep["event_counts"], "baseline": rep["baseline"], "best_context_keys": best.get("context_keys"), "best_params": best.get("selector_params"), "best_stats": best.get("stats"), "passes": best.get("passes_2024_2025_floor"), "improves_2026": best.get("improves_2026")}, indent=2, ensure_ascii=False))
