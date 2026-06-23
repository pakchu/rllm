"""Causal price-action event feature scan.

This moves away from numeric ridge action labels and toward symbolic market
structure events that an LLM can consume as compact state:
- breakout continuation candidates;
- failed breakouts / liquidity sweeps;
- range reclaims.

All event features at row t use only OHLCV rows at or before t.  Rolling range
levels are shifted by one bar, so a current-bar breakout/sweep is compared with
the prior window, not a window that already includes the breakout bar.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.alpha_linear_combo_scan import _forward_return, _load_market, _parse_list
from training.strict_bar_backtest import BarExecutionConfig, _drawdown_from_trough, _trade_stats


@dataclass(frozen=True)
class PriceActionEventScanCfg:
    input_csv: str
    output: str
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31 23:59:59"
    test_start: str = "2024-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    windows: str = "36,72,144,288,576,2016"
    horizons: str = "36,72,144,288"
    entry_delay_bars: int = 1
    min_train_events: int = 50
    min_eval_events: int = 10
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    top_k: int = 60


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den = np.asarray(den, dtype=float)
    out = np.asarray(num, dtype=float) / np.where(np.abs(den) > 1e-12, den, np.nan)
    return np.where(np.isfinite(out), out, 0.0)


def build_price_action_event_features(market: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Build symbolic/continuous causal price-action event state.

    Prior range levels use `.shift(1).rolling(w)`, so current high/low/close can
    trigger events without contaminating the range being broken or swept.
    """
    high = market["high"].astype(float)
    low = market["low"].astype(float)
    open_ = market["open"].astype(float)
    close = market["close"].astype(float)
    volume = market["volume"].astype(float) if "volume" in market.columns else pd.Series(np.ones(len(market)), index=market.index)
    body = (close - open_).to_numpy(dtype=float)
    candle_range = (high - low).replace(0.0, np.nan).to_numpy(dtype=float)
    out: dict[str, np.ndarray] = {}
    for w in windows:
        w = int(w)
        prior_high = high.shift(1).rolling(w, min_periods=w).max()
        prior_low = low.shift(1).rolling(w, min_periods=w).min()
        prior_mid = (prior_high + prior_low) / 2.0
        prior_range = (prior_high - prior_low).replace(0.0, np.nan)
        vol_min_periods = min(w, max(2, min(w, 36)))
        vol_med = volume.shift(1).rolling(w, min_periods=vol_min_periods).median().replace(0.0, np.nan)

        ph = prior_high.to_numpy(dtype=float)
        pl = prior_low.to_numpy(dtype=float)
        pm = prior_mid.to_numpy(dtype=float)
        pr = prior_range.to_numpy(dtype=float)
        h = high.to_numpy(dtype=float)
        l = low.to_numpy(dtype=float)
        o = open_.to_numpy(dtype=float)
        c = close.to_numpy(dtype=float)
        v = volume.to_numpy(dtype=float)
        vm = vol_med.to_numpy(dtype=float)

        valid = np.isfinite(ph) & np.isfinite(pl) & (pr > 0)
        break_above = valid & (c > ph)
        break_below = valid & (c < pl)
        wick_above = valid & (h > ph)
        wick_below = valid & (l < pl)
        high_sweep_reject = wick_above & (c < ph)
        low_sweep_reclaim = wick_below & (c > pl)
        failed_breakout_short = wick_above & (c < ph) & (c < o)
        failed_breakdown_long = wick_below & (c > pl) & (c > o)
        reclaim_mid_from_below = valid & (o < pm) & (c > pm) & (l <= pm)
        reject_mid_from_above = valid & (o > pm) & (c < pm) & (h >= pm)
        inside_range = valid & (h <= ph) & (l >= pl)
        outside_range_close_back_inside = valid & ((h > ph) | (l < pl)) & (c <= ph) & (c >= pl)
        volume_expansion = valid & (v > 1.5 * vm)

        prefix = f"pae_w{w}"
        out[f"{prefix}_break_above"] = break_above.astype(float)
        out[f"{prefix}_break_below"] = break_below.astype(float)
        out[f"{prefix}_high_sweep_reject"] = high_sweep_reject.astype(float)
        out[f"{prefix}_low_sweep_reclaim"] = low_sweep_reclaim.astype(float)
        out[f"{prefix}_failed_breakout_short"] = failed_breakout_short.astype(float)
        out[f"{prefix}_failed_breakdown_long"] = failed_breakdown_long.astype(float)
        out[f"{prefix}_reclaim_mid_from_below"] = reclaim_mid_from_below.astype(float)
        out[f"{prefix}_reject_mid_from_above"] = reject_mid_from_above.astype(float)
        out[f"{prefix}_inside_range"] = inside_range.astype(float)
        out[f"{prefix}_outside_close_back_inside"] = outside_range_close_back_inside.astype(float)
        out[f"{prefix}_volume_expansion"] = volume_expansion.astype(float)
        out[f"{prefix}_break_above_with_volume"] = (break_above & volume_expansion).astype(float)
        out[f"{prefix}_break_below_with_volume"] = (break_below & volume_expansion).astype(float)
        out[f"{prefix}_high_sweep_reject_with_volume"] = (high_sweep_reject & volume_expansion).astype(float)
        out[f"{prefix}_low_sweep_reclaim_with_volume"] = (low_sweep_reclaim & volume_expansion).astype(float)

        out[f"{prefix}_range_pos"] = np.clip(_safe_div(c - pl, pr), -2.0, 3.0)
        out[f"{prefix}_close_dist_prior_high"] = np.clip(_safe_div(c - ph, c), -0.5, 0.5)
        out[f"{prefix}_close_dist_prior_low"] = np.clip(_safe_div(c - pl, c), -0.5, 0.5)
        out[f"{prefix}_upper_sweep_depth"] = np.clip(_safe_div(h - ph, pr), 0.0, 5.0) * wick_above.astype(float)
        out[f"{prefix}_lower_sweep_depth"] = np.clip(_safe_div(pl - l, pr), 0.0, 5.0) * wick_below.astype(float)
        out[f"{prefix}_body_to_range"] = np.clip(_safe_div(body, candle_range), -1.0, 1.0)
    return pd.DataFrame(out, index=market.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _normal_p_value_from_t(t: float) -> float:
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(float(t)) / math.sqrt(2.0)))))


def _event_stats(events: np.ndarray, fwd: np.ndarray, mask: np.ndarray, *, min_events: int) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool) & (events > 0.5) & np.isfinite(fwd)
    y = np.asarray(fwd[valid], dtype=float)
    n = int(len(y))
    if n < int(min_events):
        return {"n": n, "mean_ret_pct": 0.0, "t_stat_like": 0.0, "p_value": 1.0, "side": "NO_TRADE"}
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 1e-12 else 0.0
    return {
        "n": n,
        "mean_ret_pct": mean * 100.0,
        "std_ret_pct": std * 100.0,
        "t_stat_like": float(t),
        "p_value": _normal_p_value_from_t(t) if se > 1e-12 else 1.0,
        "side": "LONG" if mean >= 0.0 else "SHORT",
    }


def _simulate_event_rule(
    *,
    market: pd.DataFrame,
    dates: pd.Series,
    events: np.ndarray,
    side: str,
    horizon: int,
    eval_start: str,
    eval_end: str,
    cfg: PriceActionEventScanCfg,
) -> dict[str, Any]:
    mask = np.asarray((dates >= pd.Timestamp(eval_start)) & (dates <= pd.Timestamp(eval_end)), dtype=bool)
    idxs = np.flatnonzero(mask)
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    signal = 1 if side == "LONG" else -1 if side == "SHORT" else 0
    exec_cfg = BarExecutionConfig(
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        drawdown_stop=1.0,
        pause_bars=0,
        monthly_loss_stop=1.0,
        entry_delay_bars=int(cfg.entry_delay_bars),
    )
    cost = (float(exec_cfg.fee_rate) + float(exec_cfg.slippage_rate)) * float(exec_cfg.leverage)
    hold_bars = max(1, int(horizon))
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    next_allowed = 0
    trade_returns: list[float] = []
    side_counts = {"LONG": 0, "SHORT": 0}
    skipped = 0
    for pos in idxs:
        pos = int(pos)
        if pos < next_allowed or signal == 0 or float(events[pos]) <= 0.5:
            continue
        entry_pos = pos + int(cfg.entry_delay_bars)
        exit_pos = entry_pos + hold_bars
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            skipped += 1
            continue
        entry_eq = eq
        side_counts["LONG" if signal > 0 else "SHORT"] += 1
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        for j in range(entry_pos, exit_pos):
            open_j = float(opens[j])
            if open_j <= 0.0:
                continue
            if signal > 0:
                adverse_ret = (float(lows[j]) - open_j) / open_j
                close_ret = (float(opens[j + 1]) - open_j) / open_j
            else:
                adverse_ret = (open_j - float(highs[j])) / open_j
                close_ret = (open_j - float(opens[j + 1])) / open_j
            adverse_eq = eq * (1.0 + float(cfg.leverage) * adverse_ret)
            max_dd = max(max_dd, _drawdown_from_trough(peak, adverse_eq))
            eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        next_allowed = exit_pos
        if eq <= 0.0:
            break
    if len(idxs) == 0:
        raise ValueError("no eval rows")
    eval_dates = dates.iloc[idxs]
    start_dt = pd.Timestamp(eval_dates.iloc[0]).to_pydatetime()
    end_dt = pd.Timestamp(eval_dates.iloc[-1]).to_pydatetime()
    years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    return {
        "period": {"start": str(eval_dates.iloc[0]), "end": str(eval_dates.iloc[-1]), "years": years},
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf"),
            "trade_entries": len(trade_returns),
            "side_counts": side_counts,
            "samples": int(len(idxs)),
            "skipped_missing_bars": skipped,
            "hold_bars": hold_bars,
            "entry_delay_bars": int(cfg.entry_delay_bars),
            "return_application": "event_trigger_actual_ohlc_bar_by_bar_strict_mdd",
        },
        "trade_stats": _trade_stats(trade_returns),
    }


def _score(row: dict[str, Any]) -> float:
    train = row["train"]
    test = row["test_backtest"]["sim"]
    test_stats = row["test_backtest"]["trade_stats"]
    eval_ = row["eval_backtest"]["sim"]
    trades = float(test.get("trade_entries", 0))
    if train.get("n", 0) < 50 or trades < 10:
        return -1e9
    return (
        float(test.get("cagr_to_strict_mdd", -999.0))
        + 0.01 * float(test.get("cagr_pct", 0.0))
        + min(1.0, trades / 100.0)
        - float(test_stats.get("p_value_mean_ret_approx", 1.0))
        + 0.1 * max(-3.0, min(3.0, float(eval_.get("cagr_to_strict_mdd", 0.0))))
    )


def run_scan(cfg: PriceActionEventScanCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    dates = pd.to_datetime(market["date"])
    windows = _parse_list(cfg.windows, int)
    horizons = _parse_list(cfg.horizons, int)
    features = build_price_action_event_features(market, windows)
    # Scan sparse symbolic events only; continuous state fields are for later LLM context.
    event_cols = [c for c in features.columns if set(np.unique(features[c].to_numpy(dtype=float))).issubset({0.0, 1.0}) and float(features[c].sum()) > 0]
    train_mask = np.asarray((dates >= pd.Timestamp(cfg.train_start)) & (dates <= pd.Timestamp(cfg.train_end)), dtype=bool)
    test_mask = np.asarray((dates >= pd.Timestamp(cfg.test_start)) & (dates <= pd.Timestamp(cfg.test_end)), dtype=bool)
    eval_mask = np.asarray((dates >= pd.Timestamp(cfg.eval_start)) & (dates <= pd.Timestamp(cfg.eval_end)), dtype=bool)
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        fwd = _forward_return(market["open"].astype(float), horizon=int(horizon), entry_delay_bars=int(cfg.entry_delay_bars))
        for col in event_cols:
            ev = features[col].to_numpy(dtype=float)
            train_stats = _event_stats(ev, fwd, train_mask, min_events=cfg.min_train_events)
            test_stats = _event_stats(ev, fwd, test_mask, min_events=1)
            eval_stats = _event_stats(ev, fwd, eval_mask, min_events=1)
            side = str(train_stats.get("side", "NO_TRADE"))
            try:
                test_bt = _simulate_event_rule(market=market, dates=dates, events=ev, side=side, horizon=int(horizon), eval_start=cfg.test_start, eval_end=cfg.test_end, cfg=cfg)
                eval_bt = _simulate_event_rule(market=market, dates=dates, events=ev, side=side, horizon=int(horizon), eval_start=cfg.eval_start, eval_end=cfg.eval_end, cfg=cfg)
                row = {
                    "event": col,
                    "horizon": int(horizon),
                    "train": train_stats,
                    "test_conditional": test_stats,
                    "eval_conditional": eval_stats,
                    "side": side,
                    "test_backtest": {"period": test_bt["period"], "sim": test_bt["sim"], "trade_stats": test_bt["trade_stats"]},
                    "eval_backtest": {"period": eval_bt["period"], "sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]},
                }
                row["score"] = _score(row)
                rows.append(row)
            except Exception as exc:
                rows.append({"event": col, "horizon": int(horizon), "train": train_stats, "test_conditional": test_stats, "eval_conditional": eval_stats, "side": side, "error": str(exc), "score": -1e9})
    ranked = sorted(rows, key=lambda r: float(r.get("score", -1e9)), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "feature_count": int(len(features.columns)),
        "event_count": int(len(event_cols)),
        "rows_scanned": int(len(rows)),
        "top": ranked[: int(cfg.top_k)],
        "all": ranked,
        "selection_protocol": "side is fit on train conditional mean only; test is diagnostic/selection; eval is untouched holdout",
        "leakage_guard": {
            "prior_range_uses_shifted_rolling_levels": True,
            "features_use_rows_at_or_before_t": True,
            "side_fit_uses_train_only": True,
            "eval_not_used_for_training": True,
            "entry_after_signal_by_bars": int(cfg.entry_delay_bars),
            "strict_mdd_includes_intrabar_adverse_excursion": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan causal price-action event features")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    for name in ("train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"):
        p.add_argument(f"--{name}", default=getattr(PriceActionEventScanCfg, name.replace("-", "_")))
    p.add_argument("--windows", default=PriceActionEventScanCfg.windows)
    p.add_argument("--horizons", default=PriceActionEventScanCfg.horizons)
    p.add_argument("--entry-delay-bars", type=int, default=PriceActionEventScanCfg.entry_delay_bars)
    p.add_argument("--min-train-events", type=int, default=PriceActionEventScanCfg.min_train_events)
    p.add_argument("--min-eval-events", type=int, default=PriceActionEventScanCfg.min_eval_events)
    p.add_argument("--leverage", type=float, default=PriceActionEventScanCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=PriceActionEventScanCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=PriceActionEventScanCfg.slippage_rate)
    p.add_argument("--top-k", type=int, default=PriceActionEventScanCfg.top_k)
    return p.parse_args()


def main() -> None:
    report = run_scan(PriceActionEventScanCfg(**vars(parse_args())))
    compact = []
    for row in report["top"][:10]:
        if "test_backtest" not in row:
            compact.append(row)
            continue
        compact.append({
            "event": row["event"],
            "horizon": row["horizon"],
            "side": row["side"],
            "train": row["train"],
            "test": row["test_backtest"]["sim"] | {"p": row["test_backtest"]["trade_stats"].get("p_value_mean_ret_approx")},
            "eval": row["eval_backtest"]["sim"] | {"p": row["eval_backtest"]["trade_stats"].get("p_value_mean_ret_approx")},
        })
    print(json.dumps({"output": report["config"]["output"], "event_count": report["event_count"], "rows_scanned": report["rows_scanned"], "top": compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
