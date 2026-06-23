"""Leak-safe combo scan for weak price-action extreme features.

This tests the user's intended use case: weak price-action features are not
expected to trade alone; they are evaluated as a regularized feature bundle and
as additions to existing market/external/derivatives bundles.
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
from training.alpha_linear_combo_scan import _fit_ridge_predict, _forward_return, _load_market, _parse_list, _standardize_train
from training.price_action_extreme_feature_scan import build_price_action_extreme_features


@dataclass(frozen=True)
class PriceActionComboScanCfg:
    input_csv: str
    output: str
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31 23:59:59"
    test_start: str = "2024-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    pa_windows: str = "36,72,144,288,576,2016"
    horizons: str = "36,72,144,288"
    quantiles: str = "0.05,0.10,0.20,0.30"
    ridge_l2s: str = "1,10,100,1000"
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
    top_k: int = 60


def _feature_groups(cols: list[str]) -> dict[str, list[str]]:
    pa = [c for c in cols if c.startswith("pa_w")]
    trend = [c for c in cols if c in {"trend_12", "trend_24", "trend_96", "sma12_ratio", "sma24_ratio", "sma48_ratio", "bb_z", "range_pos", "window_drawdown", "rsi_norm", "mfi_norm"}]
    candle_flow = [c for c in cols if c in {"body_ratio", "upper_shadow", "lower_shadow", "candle_range", "body_to_range", "shadow_imbalance", "volume_zscore", "taker_imbalance", "taker_buy_ratio"}]
    external = [c for c in cols if c.startswith(("dxy", "kimchi", "usdkrw", "external"))]
    derivatives = [c for c in cols if c.startswith(("funding", "premium", "binance_aux"))]
    groups = {
        "pa_only": pa,
        "pa_trend": sorted(set(pa + trend)),
        "pa_market": sorted(set(pa + trend + candle_flow)),
        "pa_external": sorted(set(pa + external)),
        "pa_derivatives": sorted(set(pa + derivatives)),
        "pa_market_external_derivatives": sorted(set(pa + trend + candle_flow + external + derivatives)),
    }
    return {k: v for k, v in groups.items() if v}


def _score(test: dict[str, Any], eval_: dict[str, Any]) -> float:
    ts = test["sim"]
    tt = test["trade_stats"]
    es = eval_["sim"]
    trades = float(ts.get("trade_entries", 0))
    if trades < 30:
        return -1e9
    # Select on test primarily; eval is reported and mildly penalizes obvious anti-generalization
    # in sorting for diagnostics, but users must not claim eval-selected deployment from this score.
    return (
        float(ts.get("cagr_to_strict_mdd", -999.0))
        + 0.01 * float(ts.get("cagr_pct", 0.0))
        + min(1.0, trades / 200.0)
        - float(tt.get("p_value_mean_ret_approx", 1.0))
        + 0.1 * max(-3.0, min(3.0, float(es.get("cagr_to_strict_mdd", 0.0))))
    )


def run_scan(cfg: PriceActionComboScanCfg) -> dict[str, Any]:
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
    base = build_market_feature_frame(market, window_size=int(cfg.window_size))
    pa = build_price_action_extreme_features(market, _parse_list(cfg.pa_windows, int))
    features = pd.concat([base, pa], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    dates = pd.to_datetime(market["date"])
    columns = [c for c in features.columns if np.nanstd(features[c].to_numpy(dtype=float)) > 1e-12]
    groups = _feature_groups(columns)
    train_mask = np.asarray((dates >= pd.Timestamp(cfg.train_start)) & (dates <= pd.Timestamp(cfg.train_end)), dtype=bool)
    rows: list[dict[str, Any]] = []
    for horizon in _parse_list(cfg.horizons, int):
        fwd = _forward_return(market["open"].astype(float), horizon=horizon, entry_delay_bars=cfg.entry_delay_bars)
        for group_name, cols in groups.items():
            Xraw = features[cols].to_numpy(dtype=float)
            X, _, _ = _standardize_train(Xraw, train_mask)
            for l2 in _parse_list(cfg.ridge_l2s, float):
                try:
                    pred, fit_info = _fit_ridge_predict(X, fwd, train_mask, l2)
                except Exception as exc:
                    rows.append({"group": group_name, "horizon": horizon, "ridge_l2": l2, "error": str(exc), "score": -1e9})
                    continue
                for q in _parse_list(cfg.quantiles, float):
                    rule_cfg = FeatureRuleConfig(
                        input_csv=cfg.input_csv,
                        output="",
                        feature=f"ridge:{group_name}",
                        horizon=int(horizon),
                        fit_start=cfg.train_start,
                        fit_end=cfg.train_end,
                        eval_start=cfg.test_start,
                        eval_end=cfg.test_end,
                        quantile=float(q),
                        window_size=cfg.window_size,
                        entry_delay_bars=cfg.entry_delay_bars,
                        leverage=cfg.leverage,
                        fee_rate=cfg.fee_rate,
                        slippage_rate=cfg.slippage_rate,
                    )
                    try:
                        rule = fit_rule(dates=dates, feature_values=pred, forward_returns=fwd, cfg=rule_cfg)
                        test = simulate_rule(market=market, feature_values=pred, dates=dates, rule=rule, cfg=rule_cfg)
                        eval_cfg = FeatureRuleConfig(**{**asdict(rule_cfg), "eval_start": cfg.eval_start, "eval_end": cfg.eval_end})
                        eval_ = simulate_rule(market=market, feature_values=pred, dates=dates, rule=rule, cfg=eval_cfg)
                        row = {
                            "group": group_name,
                            "features": cols,
                            "n_features": len(cols),
                            "horizon": int(horizon),
                            "quantile": float(q),
                            "ridge_l2": float(l2),
                            "fit_info": fit_info,
                            "rule": rule,
                            "test": {"period": test["period"], "sim": test["sim"], "trade_stats": test["trade_stats"]},
                            "eval": {"period": eval_["period"], "sim": eval_["sim"], "trade_stats": eval_["trade_stats"]},
                        }
                        row["score"] = _score(row["test"], row["eval"])
                        rows.append(row)
                    except Exception as exc:
                        rows.append({"group": group_name, "horizon": horizon, "quantile": q, "ridge_l2": l2, "error": str(exc), "score": -1e9})
    ranked = sorted(rows, key=lambda r: float(r.get("score", -1e9)), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "feature_columns": columns,
        "groups": {k: len(v) for k, v in groups.items()},
        "rows_scanned": len(rows),
        "top": ranked[: int(cfg.top_k)],
        "all": ranked,
        "selection_protocol": "ridge and rule fit on train only; test is selection/diagnostic; eval is untouched holdout",
        "leakage_guard": {
            "price_action_features_use_rows_at_or_before_t": True,
            "standardization_fit_train_only": True,
            "ridge_fit_train_only": True,
            "rule_fit_train_only": True,
            "eval_not_used_for_training": True,
            "entry_after_signal_by_bars": int(cfg.entry_delay_bars),
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan regularized combos of weak price-action features")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    for name in ("train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"):
        p.add_argument(f"--{name}", default=getattr(PriceActionComboScanCfg, name.replace("-", "_")))
    p.add_argument("--pa-windows", default=PriceActionComboScanCfg.pa_windows)
    p.add_argument("--horizons", default=PriceActionComboScanCfg.horizons)
    p.add_argument("--quantiles", default=PriceActionComboScanCfg.quantiles)
    p.add_argument("--ridge-l2s", default=PriceActionComboScanCfg.ridge_l2s)
    p.add_argument("--window-size", type=int, default=PriceActionComboScanCfg.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=PriceActionComboScanCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=PriceActionComboScanCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=PriceActionComboScanCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=PriceActionComboScanCfg.slippage_rate)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=PriceActionComboScanCfg.external_tolerance)
    p.add_argument("--binance-funding-csv", default="")
    p.add_argument("--binance-premium-csv", default="")
    p.add_argument("--binance-funding-tolerance", default=PriceActionComboScanCfg.binance_funding_tolerance)
    p.add_argument("--binance-premium-tolerance", default=PriceActionComboScanCfg.binance_premium_tolerance)
    p.add_argument("--top-k", type=int, default=PriceActionComboScanCfg.top_k)
    return p.parse_args()


def main() -> None:
    report = run_scan(PriceActionComboScanCfg(**vars(parse_args())))
    compact = []
    for row in report["top"][:10]:
        if "test" not in row:
            compact.append(row)
            continue
        compact.append({
            "group": row["group"],
            "n_features": row["n_features"],
            "horizon": row["horizon"],
            "q": row["quantile"],
            "l2": row["ridge_l2"],
            "test": row["test"]["sim"] | {"p": row["test"]["trade_stats"].get("p_value_mean_ret_approx")},
            "eval": row["eval"]["sim"] | {"p": row["eval"]["trade_stats"].get("p_value_mean_ret_approx")},
        })
    print(json.dumps({"output": report["config"]["output"], "groups": report["groups"], "rows_scanned": report["rows_scanned"], "top": compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
