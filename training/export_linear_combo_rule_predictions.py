"""Export frozen linear-combo alpha rules as live-style prediction JSONL.

The exporter refits only on the configured train window, freezes the linear score
and quantile rule, then emits timestamped TAKE/SKIP-style rows that can be fed to
``training.online_risk_overlay_backtest``.  It is intentionally thin glue around
``alpha_linear_combo_scan`` so the strict overlay evaluator can audit the same
candidate without reimplementing execution logic.
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

from preprocessing.market_features import build_market_feature_frame
from training.alpha_feature_backtest import FeatureRuleConfig, _signal_for_value, fit_rule
from training.alpha_linear_combo_scan import (
    _feature_groups,
    _fit_ridge_predict,
    _forward_return,
    _standardize_train,
)


@dataclass(frozen=True)
class ExportLinearComboRuleConfig:
    input_csv: str
    output: str
    summary_output: str = ""
    train_start: str = "2020-01-01"
    train_end: str = "2024-06-30 23:59:59"
    pred_start: str = "2024-07-01"
    pred_end: str = "2025-12-31 23:59:59"
    group: str = "external"
    horizon: int = 288
    quantile: float = 0.05
    variant: str = "original"
    window_size: int = 144
    entry_delay_bars: int = 1
    ridge_l2: float = 10.0


def load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def invert_rule(rule: dict[str, Any]) -> dict[str, Any]:
    out = dict(rule)
    out["high_side"] = "SHORT" if str(rule.get("high_side")).upper() == "LONG" else "LONG"
    out["low_side"] = "SHORT" if str(rule.get("low_side")).upper() == "LONG" else "LONG"
    return out


def _prediction_for_signal(signal: int, horizon: int) -> tuple[dict[str, Any], str]:
    if signal > 0:
        return {
            "gate": "TRADE",
            "side": "LONG",
            "hold_bars": int(horizon),
            "family": "linear_combo_rule",
            "confidence": "MEDIUM",
        }, "LONG"
    if signal < 0:
        return {
            "gate": "TRADE",
            "side": "SHORT",
            "hold_bars": int(horizon),
            "family": "linear_combo_rule",
            "confidence": "MEDIUM",
        }, "SHORT"
    return {
        "gate": "NO_TRADE",
        "side": "NONE",
        "hold_bars": 0,
        "family": "linear_combo_rule",
        "confidence": "LOW",
    }, "NO_TRADE"


def run(cfg: ExportLinearComboRuleConfig) -> dict[str, Any]:
    market = load_market(cfg.input_csv)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    dates = pd.to_datetime(market["date"])
    columns = [
        col for col in features.columns
        if np.nanstd(features[col].to_numpy(dtype=float)) > 1e-12
    ]
    groups = _feature_groups(columns)
    if cfg.group not in groups:
        raise ValueError(f"unknown group {cfg.group}; available={sorted(groups)}")
    cols = groups[cfg.group]

    train_mask = np.asarray(
        (dates >= pd.Timestamp(cfg.train_start)) & (dates <= pd.Timestamp(cfg.train_end)),
        dtype=bool,
    )
    x_raw = features[cols].to_numpy(dtype=float)
    x_std, _, _ = _standardize_train(x_raw, train_mask)
    forward_returns = _forward_return(
        market["open"].astype(float),
        horizon=int(cfg.horizon),
        entry_delay_bars=int(cfg.entry_delay_bars),
    )
    score_values, fit_info = _fit_ridge_predict(x_std, forward_returns, train_mask, float(cfg.ridge_l2))

    rule_cfg = FeatureRuleConfig(
        input_csv=cfg.input_csv,
        output="",
        feature="linear_combo",
        horizon=int(cfg.horizon),
        fit_start=cfg.train_start,
        fit_end=cfg.train_end,
        eval_start=cfg.pred_start,
        eval_end=cfg.pred_end,
        quantile=float(cfg.quantile),
        window_size=int(cfg.window_size),
        entry_delay_bars=int(cfg.entry_delay_bars),
    )
    rule = fit_rule(dates=dates, feature_values=score_values, forward_returns=forward_returns, cfg=rule_cfg)
    if cfg.variant == "inverted":
        rule = invert_rule(rule)
    elif cfg.variant != "original":
        raise ValueError("variant must be original|inverted")

    pred_mask = np.asarray(
        (dates >= pd.Timestamp(cfg.pred_start)) & (dates <= pd.Timestamp(cfg.pred_end)),
        dtype=bool,
    )
    counts = {"LONG": 0, "SHORT": 0, "NO_TRADE": 0}
    rows = []
    for pos in np.flatnonzero(pred_mask):
        signal = _signal_for_value(float(score_values[pos]), rule)
        prediction, count_key = _prediction_for_signal(signal, int(cfg.horizon))
        counts[count_key] += 1
        rows.append({
            "date": str(dates.iloc[pos]),
            "signal_pos": int(pos),
            "prediction": prediction,
            "position_scale": 1.0,
            "score": float(score_values[pos]),
            "group": cfg.group,
            "variant": cfg.variant,
        })

    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    )
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "features": cols,
        "fit_info": fit_info,
        "rule": rule,
        "rows": len(rows),
        "counts": counts,
        "leakage_guard": {
            "linear_model_fit_train_only": True,
            "thresholds_fit_train_only": True,
            "pred_period_after_train": pd.Timestamp(cfg.pred_start) > pd.Timestamp(cfg.train_end),
        },
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export linear-combo rule predictions")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--train-start", default=ExportLinearComboRuleConfig.train_start)
    parser.add_argument("--train-end", default=ExportLinearComboRuleConfig.train_end)
    parser.add_argument("--pred-start", default=ExportLinearComboRuleConfig.pred_start)
    parser.add_argument("--pred-end", default=ExportLinearComboRuleConfig.pred_end)
    parser.add_argument("--group", default=ExportLinearComboRuleConfig.group)
    parser.add_argument("--horizon", type=int, default=ExportLinearComboRuleConfig.horizon)
    parser.add_argument("--quantile", type=float, default=ExportLinearComboRuleConfig.quantile)
    parser.add_argument("--variant", choices=["original", "inverted"], default=ExportLinearComboRuleConfig.variant)
    parser.add_argument("--window-size", type=int, default=ExportLinearComboRuleConfig.window_size)
    parser.add_argument("--entry-delay-bars", type=int, default=ExportLinearComboRuleConfig.entry_delay_bars)
    parser.add_argument("--ridge-l2", type=float, default=ExportLinearComboRuleConfig.ridge_l2)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(ExportLinearComboRuleConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
