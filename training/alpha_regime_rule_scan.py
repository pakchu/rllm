"""Leak-safe regime-conditioned feature-rule scan.

Searches interactions of the form:
  if regime_feature is in its train-window low/high bucket, then trade a
  signal_feature quantile rule.

The regime bucket, signal thresholds, and long/short direction are all fit on
train only. Candidates are ranked on test and eval is an untouched audit.
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
from training.alpha_feature_backtest import FeatureRuleConfig, fit_rule, simulate_rule


@dataclass(frozen=True)
class RegimeRuleScanConfig:
    input_csv: str
    output: str
    train_start: str = "2023-01-01"
    train_end: str = "2024-06-30 23:59:59"
    test_start: str = "2024-07-01"
    test_end: str = "2025-08-31 23:59:59"
    eval_start: str = "2025-09-01"
    eval_end: str = "2026-02-28 23:59:59"
    horizons: str = "144,288"
    signal_quantile: float = 0.2
    regime_quantile: float = 0.33
    window_size: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    binance_funding_csv: str = ""
    binance_premium_csv: str = ""
    binance_funding_tolerance: str = "12h"
    binance_premium_tolerance: str = "2h"
    top_k: int = 50


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _forward_return(open_: pd.Series, *, horizon: int, entry_delay_bars: int) -> np.ndarray:
    entry = open_.shift(-int(entry_delay_bars))
    exit_ = open_.shift(-(int(entry_delay_bars) + int(horizon)))
    return ((exit_ - entry) / entry.replace(0.0, np.nan)).to_numpy(dtype=float)


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _score(report: dict[str, Any]) -> float:
    sim = report["sim"]
    stats = report["trade_stats"]
    trades = float(sim["trade_entries"])
    if trades < 30:
        return -1e9
    return float(sim["cagr_to_strict_mdd"]) + min(1.0, trades / 150.0) - float(stats.get("p_value_mean_ret_approx", 1.0))


def _candidate_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if np.nanstd(features[c].to_numpy(dtype=float)) > 1e-12]


def _default_regime_columns(cols: list[str]) -> list[str]:
    preferred = [
        "range_vol", "trend_96", "window_drawdown", "range_pos", "volume_zscore",
        "kimchi_premium_zscore", "kimchi_premium_change", "dxy_zscore", "dxy_momentum",
        "funding_zscore", "funding_rate", "premium_index_zscore", "premium_index_change",
    ]
    return [c for c in preferred if c in cols]


def run_scan(cfg: RegimeRuleScanConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    if cfg.binance_funding_csv or cfg.binance_premium_csv:
        market = attach_binance_um_aux_features(
            market,
            funding_csv=cfg.binance_funding_csv or None,
            premium_csv=cfg.binance_premium_csv or None,
            funding_tolerance=cfg.binance_funding_tolerance,
            premium_tolerance=cfg.binance_premium_tolerance,
        )
    features = build_market_feature_frame(market, window_size=cfg.window_size)
    dates = pd.to_datetime(market["date"])
    cols = _candidate_columns(features)
    regime_cols = _default_regime_columns(cols)
    train_mask = np.asarray((dates >= pd.Timestamp(cfg.train_start)) & (dates <= pd.Timestamp(cfg.train_end)), dtype=bool)

    base_cfg = FeatureRuleConfig(
        input_csv=cfg.input_csv, output="", feature="regime_signal", horizon=1,
        fit_start=cfg.train_start, fit_end=cfg.train_end, eval_start=cfg.test_start, eval_end=cfg.test_end,
        quantile=cfg.signal_quantile, window_size=cfg.window_size, entry_delay_bars=cfg.entry_delay_bars,
        leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate,
        wave_trading_root=cfg.wave_trading_root, external_tolerance=cfg.external_tolerance,
        binance_funding_csv=cfg.binance_funding_csv, binance_premium_csv=cfg.binance_premium_csv,
        binance_funding_tolerance=cfg.binance_funding_tolerance, binance_premium_tolerance=cfg.binance_premium_tolerance,
    )
    rows: list[dict[str, Any]] = []
    for horizon in _parse_ints(cfg.horizons):
        fwd = _forward_return(market["open"].astype(float), horizon=horizon, entry_delay_bars=cfg.entry_delay_bars)
        for regime_col in regime_cols:
            rv = features[regime_col].to_numpy(dtype=float)
            train_rv = rv[train_mask & np.isfinite(rv)]
            if len(train_rv) < 100:
                continue
            low_thr = float(np.quantile(train_rv, cfg.regime_quantile))
            high_thr = float(np.quantile(train_rv, 1.0 - cfg.regime_quantile))
            regimes = [("low", rv <= low_thr, low_thr), ("high", rv >= high_thr, high_thr)]
            for regime_side, regime_mask, regime_thr in regimes:
                if int((train_mask & regime_mask).sum()) < 200:
                    continue
                for signal_col in cols:
                    if signal_col == regime_col:
                        continue
                    sv = features[signal_col].to_numpy(dtype=float)
                    gated = np.where(regime_mask, sv, np.nan)
                    cfg_dict = asdict(base_cfg)
                    cfg_dict.update({"horizon": horizon, "quantile": cfg.signal_quantile, "eval_start": cfg.test_start, "eval_end": cfg.test_end})
                    test_cfg = FeatureRuleConfig(**cfg_dict)
                    try:
                        rule = fit_rule(dates=dates, feature_values=gated, forward_returns=fwd, cfg=test_cfg)
                    except Exception as exc:
                        rows.append({"horizon": horizon, "regime_col": regime_col, "regime_side": regime_side, "signal_col": signal_col, "error": str(exc)})
                        continue
                    test = simulate_rule(market=market, feature_values=gated, dates=dates, rule=rule, cfg=test_cfg)
                    eval_dict = dict(cfg_dict)
                    eval_dict.update({"eval_start": cfg.eval_start, "eval_end": cfg.eval_end})
                    eval_cfg = FeatureRuleConfig(**eval_dict)
                    eval_report = simulate_rule(market=market, feature_values=gated, dates=dates, rule=rule, cfg=eval_cfg)
                    rows.append({
                        "horizon": horizon,
                        "regime_col": regime_col,
                        "regime_side": regime_side,
                        "regime_threshold": regime_thr,
                        "signal_col": signal_col,
                        "rule": rule,
                        "test": test,
                        "eval": eval_report,
                        "test_score": _score(test),
                    })
    ranked = sorted(rows, key=lambda r: float(r.get("test_score", -1e9)), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "feature_columns": cols,
        "regime_columns": regime_cols,
        "selection_protocol": "regime bucket, signal rule, direction, thresholds fit on train; rank by test; eval audit only",
        "top_by_test": ranked[: cfg.top_k],
        "all_count": len(rows),
        "leakage_guard": {
            "regime_bucket_signal_rule_and_direction_fit_on_train_only": True,
            "test_used_for_selection_only": True,
            "eval_not_used_for_selection": True,
            "external_join": "backward_asof_no_future" if cfg.wave_trading_root else "disabled",
            "binance_aux_join": "backward_asof_no_future" if (cfg.binance_funding_csv or cfg.binance_premium_csv) else "disabled",
            "premium_index_uses_close_time_when_available": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leak-safe regime-conditioned feature rule scan")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-start", default="2023-01-01")
    p.add_argument("--train-end", default="2024-06-30 23:59:59")
    p.add_argument("--test-start", default="2024-07-01")
    p.add_argument("--test-end", default="2025-08-31 23:59:59")
    p.add_argument("--eval-start", default="2025-09-01")
    p.add_argument("--eval-end", default="2026-02-28 23:59:59")
    p.add_argument("--horizons", default="144,288")
    p.add_argument("--signal-quantile", type=float, default=0.2)
    p.add_argument("--regime-quantile", type=float, default=0.33)
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default="30min")
    p.add_argument("--binance-funding-csv", default="")
    p.add_argument("--binance-premium-csv", default="")
    p.add_argument("--binance-funding-tolerance", default=RegimeRuleScanConfig.binance_funding_tolerance)
    p.add_argument("--binance-premium-tolerance", default=RegimeRuleScanConfig.binance_premium_tolerance)
    p.add_argument("--top-k", type=int, default=50)
    return p.parse_args()


def main() -> None:
    report = run_scan(RegimeRuleScanConfig(**vars(parse_args())))
    for row in report["top_by_test"][:10]:
        if "test" not in row:
            print(json.dumps(row, ensure_ascii=False))
            continue
        ts, tt = row["test"]["sim"], row["test"]["trade_stats"]
        es, et = row["eval"]["sim"], row["eval"]["trade_stats"]
        print(json.dumps({
            "regime": f"{row['regime_col']}:{row['regime_side']}", "signal": row["signal_col"], "h": row["horizon"],
            "test": {"cagr": ts["cagr_pct"], "mdd": ts["strict_mdd_pct"], "ratio": ts["cagr_to_strict_mdd"], "trades": ts["trade_entries"], "mean": tt["mean_trade_ret_pct"], "p": tt["p_value_mean_ret_approx"]},
            "eval": {"cagr": es["cagr_pct"], "mdd": es["strict_mdd_pct"], "ratio": es["cagr_to_strict_mdd"], "trades": es["trade_entries"], "mean": et["mean_trade_ret_pct"], "p": et["p_value_mean_ret_approx"]},
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
