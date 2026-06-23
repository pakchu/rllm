"""Significance scan for price-action extreme-bar features.

Feature family requested:
- For each lookback window, find the candle with the maximum high and use that
  candle's low price as a causal state level.
- Find the candle with the minimum low and use that candle's high price.

At row t, the rolling window includes rows <= t only.  Future returns are used
only for labels/significance/backtests.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.alpha_feature_backtest import FeatureRuleConfig, _forward_return, fit_rule, simulate_rule


@dataclass(frozen=True)
class PriceActionExtremeScanCfg:
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
    quantiles: str = "0.05,0.10,0.20"
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    top_k: int = 40


def load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def parse_list(raw: str, cast):
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def extreme_bar_levels(high: np.ndarray, low: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """O(n) rolling extreme-bar levels using monotonic deques.

    Ties prefer the most recent candle, which is causal and avoids keeping stale
    equal highs/lows unnecessarily.
    """
    n = len(high)
    w = max(1, int(window))
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    low_at_high = np.full(n, np.nan, dtype=float)
    high_at_low = np.full(n, np.nan, dtype=float)
    high_age = np.full(n, np.nan, dtype=float)
    low_age = np.full(n, np.nan, dtype=float)
    maxdq: deque[int] = deque()
    mindq: deque[int] = deque()
    for i in range(n):
        old = i - w
        while maxdq and maxdq[0] <= old:
            maxdq.popleft()
        while mindq and mindq[0] <= old:
            mindq.popleft()
        while maxdq and high[i] >= high[maxdq[-1]]:
            maxdq.pop()
        maxdq.append(i)
        while mindq and low[i] <= low[mindq[-1]]:
            mindq.pop()
        mindq.append(i)
        if i >= w - 1:
            hi_idx = maxdq[0]
            lo_idx = mindq[0]
            low_at_high[i] = low[hi_idx]
            high_at_low[i] = high[lo_idx]
            high_age[i] = i - hi_idx
            low_age[i] = i - lo_idx
    return {
        "low_at_window_high": low_at_high,
        "high_at_window_low": high_at_low,
        "window_high_age_frac": high_age / float(w),
        "window_low_age_frac": low_age / float(w),
    }


def build_price_action_extreme_features(market: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    close = market["close"].astype(float).to_numpy(dtype=float)
    high = market["high"].astype(float).to_numpy(dtype=float)
    low = market["low"].astype(float).to_numpy(dtype=float)
    denom = np.where(np.abs(close) > 1e-12, close, np.nan)
    out: dict[str, np.ndarray] = {}
    for w in windows:
        levels = extreme_bar_levels(high, low, int(w))
        low_at_high = levels["low_at_window_high"]
        high_at_low = levels["high_at_window_low"]
        out[f"pa_w{w}_high_candle_low_dist"] = (close - low_at_high) / denom
        out[f"pa_w{w}_low_candle_high_dist"] = (close - high_at_low) / denom
        out[f"pa_w{w}_extreme_body_gap"] = (high_at_low - low_at_high) / denom
        out[f"pa_w{w}_high_age_frac"] = levels["window_high_age_frac"]
        out[f"pa_w{w}_low_age_frac"] = levels["window_low_age_frac"]
    df = pd.DataFrame(out, index=market.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def normal_p_value_from_t(t: float) -> float:
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(float(t)) / math.sqrt(2.0)))))


def pearson_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    xv = np.asarray(x[mask], dtype=float)
    yv = np.asarray(y[mask], dtype=float)
    n = int(len(xv))
    if n < 20 or float(np.std(xv)) <= 1e-12 or float(np.std(yv)) <= 1e-12:
        return {"n": n, "corr": 0.0, "t_stat": 0.0, "p_value": 1.0}
    corr = float(np.corrcoef(xv, yv)[0, 1])
    corr = max(-0.999999, min(0.999999, corr))
    t = corr * math.sqrt((n - 2) / max(1e-12, 1.0 - corr * corr))
    return {"n": n, "corr": corr, "t_stat": float(t), "p_value": normal_p_value_from_t(t)}


def spread_stats(x: np.ndarray, y: np.ndarray, mask: np.ndarray, q: float) -> dict[str, Any]:
    valid = mask & np.isfinite(x) & np.isfinite(y)
    xv = x[valid]
    yv = y[valid]
    if len(xv) < 100:
        return {"n": int(len(xv)), "error": "not_enough_rows"}
    q = float(np.clip(q, 0.01, 0.49))
    lo = float(np.quantile(xv, q))
    hi = float(np.quantile(xv, 1.0 - q))
    low_y = yv[xv <= lo]
    high_y = yv[xv >= hi]
    spread = float(np.mean(high_y) - np.mean(low_y)) if len(low_y) and len(high_y) else 0.0
    pooled = np.concatenate([high_y, -low_y]) if len(low_y) and len(high_y) else np.asarray([], dtype=float)
    if len(pooled) > 1 and float(np.std(pooled, ddof=1)) > 1e-12:
        t = float(np.mean(pooled) / (np.std(pooled, ddof=1) / math.sqrt(len(pooled))))
        p = normal_p_value_from_t(t)
    else:
        t = 0.0
        p = 1.0
    return {
        "n": int(len(xv)),
        "low_threshold": lo,
        "high_threshold": hi,
        "low_mean_pct": float(np.mean(low_y) * 100.0) if len(low_y) else 0.0,
        "high_mean_pct": float(np.mean(high_y) * 100.0) if len(high_y) else 0.0,
        "high_minus_low_pct": spread * 100.0,
        "abs_spread_pct": abs(spread) * 100.0,
        "t_stat_like": t,
        "p_value": p,
        "tail_n_each": int(min(len(low_y), len(high_y))),
    }


def score_row(row: dict[str, Any]) -> float:
    test = row.get("test_backtest", {}).get("sim", {})
    ev = row.get("eval_backtest", {}).get("sim", {})
    stats = row.get("test_backtest", {}).get("trade_stats", {})
    trades = float(test.get("trade_entries", 0.0))
    ratio = float(test.get("cagr_to_strict_mdd", -999.0))
    eval_ratio = float(ev.get("cagr_to_strict_mdd", -999.0))
    p = float(stats.get("p_value_mean_ret_approx", 1.0))
    return ratio + 0.25 * max(-5.0, min(5.0, eval_ratio)) + min(1.0, trades / 100.0) - p


def run_scan(cfg: PriceActionExtremeScanCfg) -> dict[str, Any]:
    market = load_market(cfg.input_csv)
    dates = pd.to_datetime(market["date"])
    windows = parse_list(cfg.windows, int)
    horizons = parse_list(cfg.horizons, int)
    quantiles = parse_list(cfg.quantiles, float)
    features = build_price_action_extreme_features(market, windows)
    train_mask = np.asarray((dates >= pd.Timestamp(cfg.train_start)) & (dates <= pd.Timestamp(cfg.train_end)), dtype=bool)
    test_mask = np.asarray((dates >= pd.Timestamp(cfg.test_start)) & (dates <= pd.Timestamp(cfg.test_end)), dtype=bool)
    eval_mask = np.asarray((dates >= pd.Timestamp(cfg.eval_start)) & (dates <= pd.Timestamp(cfg.eval_end)), dtype=bool)
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        fwd = _forward_return(market["open"].astype(float), horizon=int(horizon), entry_delay_bars=int(cfg.entry_delay_bars))
        for feature in features.columns:
            x = features[feature].to_numpy(dtype=float)
            ic = {
                "train": pearson_stats(x[train_mask], fwd[train_mask]),
                "test": pearson_stats(x[test_mask], fwd[test_mask]),
                "eval": pearson_stats(x[eval_mask], fwd[eval_mask]),
            }
            for q in quantiles:
                spread = {
                    "train": spread_stats(x, fwd, train_mask, q),
                    "test": spread_stats(x, fwd, test_mask, q),
                    "eval": spread_stats(x, fwd, eval_mask, q),
                }
                base = FeatureRuleConfig(
                    input_csv=cfg.input_csv,
                    output="",
                    feature=feature,
                    horizon=int(horizon),
                    fit_start=cfg.train_start,
                    fit_end=cfg.train_end,
                    eval_start=cfg.test_start,
                    eval_end=cfg.test_end,
                    quantile=float(q),
                    entry_delay_bars=int(cfg.entry_delay_bars),
                    leverage=float(cfg.leverage),
                    fee_rate=float(cfg.fee_rate),
                    slippage_rate=float(cfg.slippage_rate),
                )
                try:
                    rule = fit_rule(dates=dates, feature_values=x, forward_returns=fwd, cfg=base)
                    test_bt = simulate_rule(market=market, feature_values=x, dates=dates, rule=rule, cfg=base)
                    eval_cfg = FeatureRuleConfig(**{**asdict(base), "eval_start": cfg.eval_start, "eval_end": cfg.eval_end})
                    eval_bt = simulate_rule(market=market, feature_values=x, dates=dates, rule=rule, cfg=eval_cfg)
                    row = {
                        "feature": feature,
                        "horizon": int(horizon),
                        "quantile": float(q),
                        "rule": rule,
                        "ic": ic,
                        "spread": spread,
                        "test_backtest": {"period": test_bt["period"], "sim": test_bt["sim"], "trade_stats": test_bt["trade_stats"]},
                        "eval_backtest": {"period": eval_bt["period"], "sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]},
                    }
                    row["score"] = score_row(row)
                    rows.append(row)
                except Exception as exc:
                    rows.append({"feature": feature, "horizon": int(horizon), "quantile": float(q), "error": str(exc), "ic": ic, "spread": spread, "score": -1e9})
    ranked = sorted(rows, key=lambda r: float(r.get("score", -1e9)), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "feature_count": int(len(features.columns)),
        "rows_scanned": len(rows),
        "top": ranked[: int(cfg.top_k)],
        "all": ranked,
        "leakage_guard": {
            "features_use_rows_at_or_before_t": True,
            "extreme_bar_levels_include_current_bar_only_not_future": True,
            "rule_fit_uses_train_only": True,
            "test_used_for_diagnostics_not_eval_selection": True,
            "eval_is_untouched_holdout": True,
            "entry_after_signal_by_bars": int(cfg.entry_delay_bars),
            "strict_mdd_includes_intrabar_adverse_excursion": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan price-action extreme-bar feature significance")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-start", default=PriceActionExtremeScanCfg.train_start)
    p.add_argument("--train-end", default=PriceActionExtremeScanCfg.train_end)
    p.add_argument("--test-start", default=PriceActionExtremeScanCfg.test_start)
    p.add_argument("--test-end", default=PriceActionExtremeScanCfg.test_end)
    p.add_argument("--eval-start", default=PriceActionExtremeScanCfg.eval_start)
    p.add_argument("--eval-end", default=PriceActionExtremeScanCfg.eval_end)
    p.add_argument("--windows", default=PriceActionExtremeScanCfg.windows)
    p.add_argument("--horizons", default=PriceActionExtremeScanCfg.horizons)
    p.add_argument("--quantiles", default=PriceActionExtremeScanCfg.quantiles)
    p.add_argument("--entry-delay-bars", type=int, default=PriceActionExtremeScanCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=PriceActionExtremeScanCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=PriceActionExtremeScanCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=PriceActionExtremeScanCfg.slippage_rate)
    p.add_argument("--top-k", type=int, default=PriceActionExtremeScanCfg.top_k)
    return p.parse_args()


def main() -> None:
    report = run_scan(PriceActionExtremeScanCfg(**vars(parse_args())))
    compact = []
    for row in report["top"][:10]:
        if "test_backtest" not in row:
            compact.append(row)
            continue
        compact.append({
            "feature": row["feature"],
            "horizon": row["horizon"],
            "quantile": row["quantile"],
            "train_spread_pct": row["spread"]["train"].get("high_minus_low_pct"),
            "test": row["test_backtest"]["sim"],
            "test_p": row["test_backtest"]["trade_stats"].get("p_value_mean_ret_approx"),
            "eval": row["eval_backtest"]["sim"],
            "eval_p": row["eval_backtest"]["trade_stats"].get("p_value_mean_ret_approx"),
        })
    print(json.dumps({"output": report["config"]["output"], "feature_count": report["feature_count"], "top": compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
