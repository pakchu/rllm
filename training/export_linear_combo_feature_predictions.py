"""Export leak-safe linear-combo feature quantile signals as action predictions.

This reuses the linear feature groups from alpha_linear_combo_scan, fits the
ridge model and quantile rule on a train window only, then emits TRADE/NO_TRADE
prediction rows for a later eval window so stricter execution overlays (TP/SL,
pauses) can be tested with online_risk_overlay_backtest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.alpha_feature_backtest import FeatureRuleConfig, _forward_return, _signal_for_value, fit_rule
from training.alpha_linear_combo_scan import _feature_groups, _fit_ridge_predict, _load_market, _standardize_train


@dataclass(frozen=True)
class ExportConfig:
    input_csv: str
    output: str
    summary_output: str
    group: str
    horizon: int
    quantile: float
    train_start: str = "2023-01-01"
    train_end: str = "2024-06-30 23:59:59"
    eval_start: str = "2024-07-01"
    eval_end: str = "2025-12-31 23:59:59"
    window_size: int = 144
    entry_delay_bars: int = 1
    ridge_l2: float = 10.0
    wave_trading_root: str = ""
    external_tolerance: str = "30min"


def export_predictions(cfg: ExportConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    dates = pd.to_datetime(market["date"])
    columns = [c for c in features.columns if np.nanstd(features[c].to_numpy(dtype=float)) > 1e-12]
    groups = _feature_groups(columns)
    if cfg.group not in groups:
        raise ValueError(f"unknown group {cfg.group}; available={sorted(groups)}")
    cols = groups[cfg.group]
    train_mask = np.asarray((dates >= pd.Timestamp(cfg.train_start)) & (dates <= pd.Timestamp(cfg.train_end)), dtype=bool)
    Xraw = features[cols].to_numpy(dtype=float)
    X, _, _ = _standardize_train(Xraw, train_mask)
    fwd = _forward_return(market["open"].astype(float), horizon=int(cfg.horizon), entry_delay_bars=int(cfg.entry_delay_bars))
    pred_values, fit_info = _fit_ridge_predict(X, fwd, train_mask, float(cfg.ridge_l2))
    rule_cfg = FeatureRuleConfig(
        input_csv=cfg.input_csv,
        output="",
        feature="linear_combo",
        horizon=int(cfg.horizon),
        fit_start=cfg.train_start,
        fit_end=cfg.train_end,
        eval_start=cfg.eval_start,
        eval_end=cfg.eval_end,
        quantile=float(cfg.quantile),
        window_size=int(cfg.window_size),
        entry_delay_bars=int(cfg.entry_delay_bars),
        leverage=0.76,
        wave_trading_root=cfg.wave_trading_root,
        external_tolerance=cfg.external_tolerance,
    )
    rule = fit_rule(dates=dates, feature_values=pred_values, forward_returns=fwd, cfg=rule_cfg)
    mask = np.asarray((dates >= pd.Timestamp(cfg.eval_start)) & (dates <= pd.Timestamp(cfg.eval_end)), dtype=bool)
    rows: list[dict[str, Any]] = []
    trade_rows = 0
    side_counts = {"LONG": 0, "SHORT": 0}
    for pos in np.flatnonzero(mask):
        sig = _signal_for_value(float(pred_values[pos]), rule)
        if sig > 0:
            pred = {"gate": "TRADE", "side": "LONG", "hold_bars": int(cfg.horizon), "family": f"linear_combo:{cfg.group}", "confidence": "HIGH"}
        elif sig < 0:
            pred = {"gate": "TRADE", "side": "SHORT", "hold_bars": int(cfg.horizon), "family": f"linear_combo:{cfg.group}", "confidence": "HIGH"}
        else:
            pred = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "linear_combo", "confidence": "HIGH"}
        if pred["gate"] == "TRADE":
            trade_rows += 1
            side_counts[pred["side"]] += 1
        rows.append({"date": str(dates.iloc[pos]), "signal_pos": int(pos), "prediction": pred, "feature_value": float(pred_values[pos])})
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")
    report = {
        "config": asdict(cfg),
        "features": cols,
        "fit_info": fit_info,
        "rule": rule,
        "rows": len(rows),
        "trade_rows": trade_rows,
        "side_counts": side_counts,
        "predictions_output": cfg.output,
        "leakage_guard": {"ridge_fit_train_only": True, "quantile_rule_fit_train_only": True, "eval_after_train": pd.Timestamp(cfg.eval_start) > pd.Timestamp(cfg.train_end), "external_join": "backward_asof_no_future" if cfg.wave_trading_root else "disabled"},
    }
    Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export linear combo feature rule predictions")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--quantile", type=float, required=True)
    p.add_argument("--train-start", default="2023-01-01")
    p.add_argument("--train-end", default="2024-06-30 23:59:59")
    p.add_argument("--eval-start", default="2024-07-01")
    p.add_argument("--eval-end", default="2025-12-31 23:59:59")
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--ridge-l2", type=float, default=10.0)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default="30min")
    return p.parse_args()


def main() -> None:
    rep = export_predictions(ExportConfig(**vars(parse_args())))
    print(json.dumps({"predictions_output": rep["predictions_output"], "rows": rep["rows"], "trade_rows": rep["trade_rows"], "side_counts": rep["side_counts"], "rule": rep["rule"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
