"""Strict backtest for univariate alpha-feature quantile rules.

The rule is fit only on a chronological fit window:
- choose top/bottom feature quantile thresholds;
- choose whether high feature values should be LONG or SHORT from fit-window
  high-minus-low forward return.

Then it trades the frozen rule on a later evaluation window with bar-by-bar
strict MDD, entry delay, costs, and non-overlapping holds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.binance_aux_features import attach_binance_um_aux_features
from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.strict_bar_backtest import BarExecutionConfig, _drawdown_from_trough, _trade_stats


@dataclass(frozen=True)
class FeatureRuleConfig:
    input_csv: str
    output: str
    feature: str
    horizon: int
    fit_start: str = "2024-01-01"
    fit_end: str = "2025-08-31 23:59:59"
    eval_start: str = "2025-09-01"
    eval_end: str = "2026-02-28 23:59:59"
    quantile: float = 0.2
    window_size: int = 144
    entry_delay_bars: int = 1
    cooldown_bars: int = 0
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    binance_funding_csv: str = ""
    binance_premium_csv: str = ""
    binance_funding_tolerance: str = "12h"
    binance_premium_tolerance: str = "2h"


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _forward_return(open_: pd.Series, *, horizon: int, entry_delay_bars: int) -> np.ndarray:
    entry = open_.shift(-int(entry_delay_bars))
    exit_ = open_.shift(-(int(entry_delay_bars) + int(horizon)))
    return ((exit_ - entry) / entry.replace(0.0, np.nan)).to_numpy(dtype=float)


def fit_rule(
    *,
    dates: pd.Series,
    feature_values: np.ndarray,
    forward_returns: np.ndarray,
    cfg: FeatureRuleConfig,
) -> dict[str, Any]:
    fit_mask = (
        (dates >= pd.Timestamp(cfg.fit_start))
        & (dates <= pd.Timestamp(cfg.fit_end))
        & np.isfinite(feature_values)
        & np.isfinite(forward_returns)
    )
    x = feature_values[np.asarray(fit_mask, dtype=bool)]
    y = forward_returns[np.asarray(fit_mask, dtype=bool)]
    if x.size < 100:
        raise ValueError(f"not enough fit samples: {x.size}")
    q = float(np.clip(cfg.quantile, 0.01, 0.49))
    low_threshold = float(np.quantile(x, q))
    high_threshold = float(np.quantile(x, 1.0 - q))
    high_mean = float(np.mean(y[x >= high_threshold])) if np.any(x >= high_threshold) else 0.0
    low_mean = float(np.mean(y[x <= low_threshold])) if np.any(x <= low_threshold) else 0.0
    # If high feature values outperformed low values in fit, high=>LONG and
    # low=>SHORT.  Otherwise invert.  This direction is frozen for eval.
    high_side = "LONG" if high_mean >= low_mean else "SHORT"
    low_side = "SHORT" if high_side == "LONG" else "LONG"
    return {
        "fit_n": int(x.size),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "fit_low_mean_pct": low_mean * 100.0,
        "fit_high_mean_pct": high_mean * 100.0,
        "fit_high_minus_low_pct": (high_mean - low_mean) * 100.0,
        "high_side": high_side,
        "low_side": low_side,
    }


def _signal_for_value(value: float, rule: dict[str, Any]) -> int:
    if not np.isfinite(value):
        return 0
    if value >= float(rule["high_threshold"]):
        return 1 if rule["high_side"] == "LONG" else -1
    if value <= float(rule["low_threshold"]):
        return 1 if rule["low_side"] == "LONG" else -1
    return 0


def simulate_rule(
    *,
    market: pd.DataFrame,
    feature_values: np.ndarray,
    dates: pd.Series,
    rule: dict[str, Any],
    cfg: FeatureRuleConfig,
) -> dict[str, Any]:
    eval_mask = (dates >= pd.Timestamp(cfg.eval_start)) & (dates <= pd.Timestamp(cfg.eval_end))
    eval_indices = np.flatnonzero(np.asarray(eval_mask, dtype=bool))
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
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
    hold_bars = max(1, int(cfg.horizon))
    next_allowed = 0
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    entries = 0
    trade_returns: list[float] = []
    executed: list[dict[str, Any]] = []
    side_counts = {"LONG": 0, "SHORT": 0}
    skipped = 0

    for pos in eval_indices:
        if int(pos) < next_allowed:
            continue
        signal = _signal_for_value(float(feature_values[pos]), rule)
        if signal == 0:
            continue
        entry_pos = int(pos) + int(cfg.entry_delay_bars)
        exit_pos = entry_pos + hold_bars
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            skipped += 1
            continue
        side_counts["LONG" if signal > 0 else "SHORT"] += 1
        entry_eq = eq
        entries += 1
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
        executed.append(
            {
                "signal_date": str(dates.iloc[pos]),
                "entry_date": str(dates.iloc[entry_pos]),
                "exit_date": str(dates.iloc[exit_pos]),
                "side": "LONG" if signal > 0 else "SHORT",
                "feature": str(cfg.feature),
                "feature_value": float(feature_values[pos]),
                "horizon": int(cfg.horizon),
                "executed_ret_pct": (eq / entry_eq - 1.0) * 100.0,
            }
        )
        next_allowed = exit_pos + max(0, int(cfg.cooldown_bars))
        if eq <= 0.0:
            break

    eval_dates = dates.iloc[eval_indices]
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
            "trade_entries": entries,
            "side_counts": side_counts,
            "samples": int(len(eval_indices)),
            "skipped_missing_bars": skipped,
            "hold_bars": hold_bars,
            "entry_delay_bars": int(cfg.entry_delay_bars),
            "return_application": "actual_ohlc_bar_by_bar_feature_quantile_strict_mdd",
        },
        "trade_stats": _trade_stats(trade_returns),
        "executed": executed,
    }


def run_feature_rule_backtest(cfg: FeatureRuleConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(
            market,
            wave_trading_root=cfg.wave_trading_root,
            tolerance=cfg.external_tolerance,
        )
    if cfg.binance_funding_csv or cfg.binance_premium_csv:
        market = attach_binance_um_aux_features(
            market,
            funding_csv=cfg.binance_funding_csv or None,
            premium_csv=cfg.binance_premium_csv or None,
            funding_tolerance=cfg.binance_funding_tolerance,
            premium_tolerance=cfg.binance_premium_tolerance,
        )
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    if cfg.feature not in features.columns:
        raise ValueError(f"feature not found: {cfg.feature}")
    dates = pd.to_datetime(market["date"])
    x = features[cfg.feature].to_numpy(dtype=float)
    fwd = _forward_return(market["open"].astype(float), horizon=int(cfg.horizon), entry_delay_bars=int(cfg.entry_delay_bars))
    rule = fit_rule(dates=dates, feature_values=x, forward_returns=fwd, cfg=cfg)
    result = simulate_rule(market=market, feature_values=x, dates=dates, rule=rule, cfg=cfg)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rule": rule,
        "result": result,
        "leakage_guard": {
            "feature_uses_rows_at_or_before_t": True,
            "rule_fit_uses_fit_window_only": True,
            "eval_window_after_fit_window": pd.Timestamp(cfg.eval_start) > pd.Timestamp(cfg.fit_end),
            "external_join": "backward_asof_no_future" if cfg.wave_trading_root else "disabled",
            "binance_aux_join": "backward_asof_no_future" if (cfg.binance_funding_csv or cfg.binance_premium_csv) else "disabled",
            "premium_index_uses_close_time_when_available": True,
            "entry_after_signal_by_bars": int(cfg.entry_delay_bars),
            "strict_mdd_includes_intrabar_adverse_excursion": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest a no-leak univariate alpha feature rule")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--feature", required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--fit-start", default="2024-01-01")
    p.add_argument("--fit-end", default="2025-08-31 23:59:59")
    p.add_argument("--eval-start", default="2025-09-01")
    p.add_argument("--eval-end", default="2026-02-28 23:59:59")
    p.add_argument("--quantile", type=float, default=0.2)
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default="30min")
    p.add_argument("--binance-funding-csv", default="")
    p.add_argument("--binance-premium-csv", default="")
    p.add_argument("--binance-funding-tolerance", default=FeatureRuleConfig.binance_funding_tolerance)
    p.add_argument("--binance-premium-tolerance", default=FeatureRuleConfig.binance_premium_tolerance)
    return p.parse_args()


def main() -> None:
    report = run_feature_rule_backtest(FeatureRuleConfig(**vars(parse_args())))
    r = report["result"]["sim"]
    t = report["result"]["trade_stats"]
    print(json.dumps({"output": report["config"]["output"], "rule": report["rule"], "sim": r, "trade_stats": t}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
