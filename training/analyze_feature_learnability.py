"""Past-only feature learnability diagnostic for trade-side labels.

This is not a trading strategy. It checks whether the current feature set can
predict executable LONG/SHORT path-outcome labels out-of-sample without adding
new dependencies or touching eval for model selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.data_sources import load_market_data
from training.path_outcome_dataset import PathOutcomeConfig, compute_trade_path_outcome


DEFAULT_FEATURES = (
    "trend_96",
    "range_vol",
    "window_drawdown",
    "range_pos",
    "bb_z",
    "rsi_norm",
    "mfi_norm",
    "volume_zscore",
    "body_ratio",
    "body_to_range",
    "upper_shadow",
    "lower_shadow",
    "return_zscore_48",
    "taker_imbalance",
    "dxy_zscore",
    "dxy_momentum",
    "kimchi_premium_zscore",
    "kimchi_premium_change",
    "usdkrw_zscore",
    "usdkrw_momentum",
)


@dataclass(frozen=True)
class Split:
    name: str
    start: str
    end: str


def _parse_feature_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_FEATURES
    return tuple(x.strip() for x in value.split(",") if x.strip())


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    return mu, sigma


def _standardize_apply(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (x - mu) / sigma
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, lr: float, steps: int, l2: float) -> tuple[np.ndarray, float]:
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    y = y.astype(np.float64)
    n = max(1, len(y))
    for _ in range(max(1, int(steps))):
        p = _sigmoid(x @ w + b)
        err = p - y
        w -= float(lr) * ((x.T @ err) / n + float(l2) * w)
        b -= float(lr) * float(np.mean(err))
    return w, b


def _metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(np.int64)
    y = y.astype(np.int64)
    acc = float(np.mean(pred == y)) if len(y) else 0.0
    long_support = int(np.sum(y == 1))
    short_support = int(np.sum(y == 0))
    long_recall = float(np.sum((pred == 1) & (y == 1)) / max(1, long_support))
    short_recall = float(np.sum((pred == 0) & (y == 0)) / max(1, short_support))
    edge = prob - 0.5
    return {
        "samples": int(len(y)),
        "accuracy": acc,
        "balanced_recall": 0.5 * (long_recall + short_recall),
        "long_recall": long_recall,
        "short_recall": short_recall,
        "target_counts": {"SHORT": short_support, "LONG": long_support},
        "pred_counts": {"SHORT": int(np.sum(pred == 0)), "LONG": int(np.sum(pred == 1))},
        "mean_prob_long": float(np.mean(prob)) if len(prob) else 0.0,
        "mean_abs_edge": float(np.mean(np.abs(edge))) if len(edge) else 0.0,
    }


def _build_rows(
    market_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...],
    window_size: int,
    split: Split,
    path_cfg: PathOutcomeConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    start_ts = pd.to_datetime(split.start)
    end_ts = pd.to_datetime(split.end)
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    dates: list[str] = []
    end_t = len(market_df) - int(path_cfg.hold_bars) - int(path_cfg.entry_delay_bars) - 1
    for t in range(max(0, int(window_size) - 1), max(0, end_t)):
        dt = pd.to_datetime(market_df.loc[t, "date"])
        if dt < start_ts or dt > end_ts:
            continue
        long_out = compute_trade_path_outcome(market_df, t, "LONG", path_cfg)
        short_out = compute_trade_path_outcome(market_df, t, "SHORT", path_cfg)
        if long_out is None or short_out is None:
            continue
        best = long_out if long_out.utility >= short_out.utility else short_out
        trade_ok = (
            float(best.utility) > float(path_cfg.hold_margin)
            and float(best.net_return) > float(path_cfg.min_net_return)
            and float(best.mae) <= float(path_cfg.max_mae)
        )
        if not trade_ok:
            continue
        row = feature_frame.iloc[t]
        x_rows.append([float(row.get(c, 0.0)) for c in feature_cols])
        y_rows.append(1 if best.side == "LONG" else 0)
        dates.append(str(dt))
    if not x_rows:
        return np.empty((0, len(feature_cols))), np.empty((0,), dtype=np.int64), []
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64), dates


def run(args: argparse.Namespace) -> dict:
    market_df = load_market_data(
        source=args.source,
        input_csv=args.input_csv,
        timeframe=args.timeframe,
        symbol=args.symbol,
        start_date=args.data_start_date,
        end_date=args.data_end_date,
        market_type=args.market_type,
    )
    if args.wave_trading_root:
        market_df = attach_wave_trading_external_features(
            market_df,
            wave_trading_root=args.wave_trading_root,
            tolerance=args.external_tolerance or None,
        )
    feature_frame = build_market_feature_frame(market_df, window_size=int(args.window_size))
    feature_cols = tuple(c for c in _parse_feature_list(args.features) if c in feature_frame.columns)
    if not feature_cols:
        raise ValueError("No requested feature columns exist in feature frame")
    path_cfg = PathOutcomeConfig(
        hold_bars=int(args.target_horizon),
        entry_delay_bars=int(args.path_entry_delay_bars),
        fee_rate=float(args.utility_fee_rate),
        slippage_rate=float(args.utility_slippage_rate),
        leverage=float(args.utility_leverage),
        mae_penalty=float(args.path_mae_penalty),
        mfe_bonus=float(args.path_mfe_bonus),
        hold_margin=float(args.utility_hold_margin),
        min_net_return=float(args.path_min_net_return),
        max_mae=float(args.path_max_mae),
    )
    splits = [
        Split("train", args.train_start_date, args.train_end_date),
        Split("test", args.test_start_date, args.test_end_date),
        Split("eval", args.eval_start_date, args.eval_end_date),
    ]
    data = {
        sp.name: _build_rows(
            market_df,
            feature_frame,
            feature_cols=feature_cols,
            window_size=int(args.window_size),
            split=sp,
            path_cfg=path_cfg,
        )
        for sp in splits
    }
    x_train, y_train, _ = data["train"]
    if len(y_train) < 10:
        raise ValueError(f"Too few train samples: {len(y_train)}")
    mu, sigma = _standardize_fit(x_train)
    x_train_z = _standardize_apply(x_train, mu, sigma)
    w, b = _fit_logistic(x_train_z, y_train, lr=float(args.lr), steps=int(args.steps), l2=float(args.l2))

    split_metrics = {}
    for name, (x, y, dates) in data.items():
        z = _standardize_apply(x, mu, sigma)
        prob = _sigmoid(z @ w + b)
        split_metrics[name] = _metrics(y, prob) | {
            "period": {"start": dates[0] if dates else None, "end": dates[-1] if dates else None},
        }
    top_idx = np.argsort(np.abs(w))[::-1][: min(12, len(w))]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "features": list(feature_cols),
        "model": {"type": "numpy_logistic_regression", "lr": float(args.lr), "steps": int(args.steps), "l2": float(args.l2)},
        "splits": split_metrics,
        "top_abs_weights": [{"feature": feature_cols[i], "weight": float(w[i])} for i in top_idx],
        "leakage_guard": {
            "uses_past_only_feature_frame": True,
            "uses_future_only_for_labels": True,
            "eval_not_used_for_training": True,
            "external_join": "backward_asof_no_future" if args.wave_trading_root else "disabled",
        },
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check if current past-only features can learn trade-side labels")
    p.add_argument("--source", default="csv")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--market-type", default="futures")
    p.add_argument("--data-start-date", default=None)
    p.add_argument("--data-end-date", default=None)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default="")
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--target-horizon", type=int, default=144)
    p.add_argument("--train-start-date", required=True)
    p.add_argument("--train-end-date", required=True)
    p.add_argument("--test-start-date", required=True)
    p.add_argument("--test-end-date", required=True)
    p.add_argument("--eval-start-date", required=True)
    p.add_argument("--eval-end-date", required=True)
    p.add_argument("--features", default="")
    p.add_argument("--utility-leverage", type=float, default=0.5)
    p.add_argument("--utility-fee-rate", type=float, default=0.0004)
    p.add_argument("--utility-slippage-rate", type=float, default=0.0001)
    p.add_argument("--utility-hold-margin", type=float, default=0.0)
    p.add_argument("--path-entry-delay-bars", type=int, default=1)
    p.add_argument("--path-mae-penalty", type=float, default=1.0)
    p.add_argument("--path-mfe-bonus", type=float, default=0.0)
    p.add_argument("--path-min-net-return", type=float, default=0.0)
    p.add_argument("--path-max-mae", type=float, default=0.03)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--l2", type=float, default=0.001)
    p.add_argument("--output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
