"""Recency-weighted symbolic action ridge experiments.

Keeps old history for regularization, but weights recent rows more heavily instead
of dropping 2020-2022 wholesale.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.symbolic_action_ridge import FeatureSpace, _candidate_rows_from_preds, choose_best, load_jsonl, target_value, write_jsonl


def _date(row: dict[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(str(row.get("date")))


def _weights(rows: list[dict[str, Any]], *, mode: str, half_life_days: float, recent_start: str, recent_weight: float) -> np.ndarray:
    mode = str(mode)
    if mode == "none":
        return np.ones(len(rows), dtype=np.float64)
    dates = np.asarray([_date(r).timestamp() / 86400.0 for r in rows], dtype=np.float64)
    if mode == "exp":
        max_day = float(np.max(dates)) if len(dates) else 0.0
        return np.power(0.5, np.maximum(0.0, max_day - dates) / max(1e-9, float(half_life_days)))
    if mode == "step":
        cutoff = pd.Timestamp(recent_start).timestamp() / 86400.0
        return np.where(dates >= cutoff, float(recent_weight), 1.0).astype(np.float64)
    raise ValueError("mode must be one of none, exp, step")


def fit_weighted_ridge(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, alpha: float) -> np.ndarray:
    w = np.sqrt(np.asarray(sample_weight, dtype=np.float64).reshape(-1))
    xw = x * w[:, None]
    yw = y * w
    reg = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 1e-9
    return np.linalg.pinv(xw.T @ xw + reg) @ xw.T @ yw


def train_predict_weighted(
    *,
    train_jsonl: str,
    eval_jsonl: str,
    predictions_output: str,
    alpha: float = 10000.0,
    threshold: float = 0.003,
    min_gap: float = 0.0,
    target: str = "net_return",
    min_feature_count: int = 5,
    weight_mode: str = "none",
    half_life_days: float = 365.0,
    recent_start: str = "2023-01-01",
    recent_weight: float = 2.0,
) -> dict[str, Any]:
    train = load_jsonl(train_jsonl)
    ev = load_jsonl(eval_jsonl)
    fs = FeatureSpace.fit(train, min_count=min_feature_count)
    x = fs.matrix(train)
    y = np.asarray([target_value(r, target=target) for r in train], dtype=np.float64)
    sw = _weights(train, mode=weight_mode, half_life_days=half_life_days, recent_start=recent_start, recent_weight=recent_weight)
    beta = fit_weighted_ridge(x, y, sw, alpha=float(alpha))
    train_pred = x @ beta
    ev_pred = fs.matrix(ev) @ beta
    chosen = choose_best(_candidate_rows_from_preds(ev, ev_pred), threshold=float(threshold), min_gap=float(min_gap))
    write_jsonl(predictions_output, chosen)
    corr = 0.0 if len(y) < 2 or np.std(y) < 1e-12 or np.std(train_pred) < 1e-12 else float(np.corrcoef(y, train_pred)[0, 1])
    return {
        "train_jsonl": train_jsonl,
        "eval_jsonl": eval_jsonl,
        "predictions_output": predictions_output,
        "config": {"alpha": alpha, "threshold": threshold, "min_gap": min_gap, "target": target, "min_feature_count": min_feature_count, "weight_mode": weight_mode, "half_life_days": half_life_days, "recent_start": recent_start, "recent_weight": recent_weight, "features": len(fs.vocab)},
        "fit": {"train_rows": len(train), "eval_rows": len(ev), "train_corr": corr, "train_rmse_pct": math.sqrt(float(np.mean((train_pred - y) ** 2))) * 100.0, "weight_min": float(np.min(sw)), "weight_max": float(np.max(sw)), "weight_mean": float(np.mean(sw))},
        "chosen_counts": {},
    }


def rolling_weighted(
    *,
    history_jsonl: str,
    eval_jsonl: str,
    predictions_output: str,
    summary_output: str,
    start_date: str,
    end_date: str,
    alpha: float = 10000.0,
    threshold: float = 0.003,
    min_gap: float = 0.0,
    target: str = "net_return",
    min_feature_count: int = 5,
    weight_mode: str = "step",
    half_life_days: float = 365.0,
    recent_start: str = "2023-01-01",
    recent_weight: float = 2.0,
) -> dict[str, Any]:
    history = load_jsonl(history_jsonl)
    ev = load_jsonl(eval_jsonl)
    all_rows = sorted(history + ev, key=_date)
    eval_rows = [r for r in ev if pd.Timestamp(start_date) <= _date(r) < pd.Timestamp(end_date)]
    months = pd.date_range(pd.Timestamp(start_date).replace(day=1), pd.Timestamp(end_date), freq="MS")
    out: list[dict[str, Any]] = []
    month_summaries = []
    for mstart in months:
        mend = mstart + pd.offsets.MonthBegin(1)
        if mstart >= pd.Timestamp(end_date):
            continue
        train = [r for r in all_rows if _date(r) < mstart]
        test = [r for r in eval_rows if mstart <= _date(r) < min(mend, pd.Timestamp(end_date))]
        if not test:
            continue
        fs = FeatureSpace.fit(train, min_count=min_feature_count)
        x = fs.matrix(train)
        y = np.asarray([target_value(r, target=target) for r in train], dtype=np.float64)
        sw = _weights(train, mode=weight_mode, half_life_days=half_life_days, recent_start=recent_start, recent_weight=recent_weight)
        beta = fit_weighted_ridge(x, y, sw, alpha=float(alpha))
        preds = fs.matrix(test) @ beta
        chosen = choose_best(_candidate_rows_from_preds(test, preds), threshold=float(threshold), min_gap=float(min_gap))
        out.extend(chosen)
        month_summaries.append({"month": str(mstart.date())[:7], "train_rows": len(train), "candidate_rows": len(test), "signals": len(chosen), "trade_signals": sum(1 for r in chosen if r["prediction"].get("gate") == "TRADE"), "features": len(fs.vocab), "weight_mean": float(np.mean(sw)), "weight_min": float(np.min(sw)), "weight_max": float(np.max(sw))})
    write_jsonl(predictions_output, out)
    rep = {"config": {"alpha": alpha, "threshold": threshold, "min_gap": min_gap, "target": target, "min_feature_count": min_feature_count, "weight_mode": weight_mode, "half_life_days": half_life_days, "recent_start": recent_start, "recent_weight": recent_weight}, "history_jsonl": history_jsonl, "eval_jsonl": eval_jsonl, "predictions_output": predictions_output, "period": {"start": start_date, "end": end_date}, "rows": len(out), "months": month_summaries, "leakage_guard": {"each_month_fit_uses_rows_before_month_start_only": True, "weights_use_dates_only_not_future_returns": True, "config_fixed_before_eval": True}}
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_output).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    return rep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weighted symbolic ridge")
    sub = p.add_subparsers(dest="cmd", required=True)
    tp = sub.add_parser("train-predict")
    tp.add_argument("--train-jsonl", required=True); tp.add_argument("--eval-jsonl", required=True); tp.add_argument("--predictions-output", required=True)
    ro = sub.add_parser("rolling")
    ro.add_argument("--history-jsonl", required=True); ro.add_argument("--eval-jsonl", required=True); ro.add_argument("--predictions-output", required=True); ro.add_argument("--summary-output", required=True); ro.add_argument("--start-date", required=True); ro.add_argument("--end-date", required=True)
    for sp in (tp, ro):
        sp.add_argument("--alpha", type=float, default=10000.0)
        sp.add_argument("--threshold", type=float, default=0.003)
        sp.add_argument("--min-gap", type=float, default=0.0)
        sp.add_argument("--target", choices=["utility", "net_return", "risk_adjusted", "tail_risk", "distributional_safety"], default="net_return")
        sp.add_argument("--min-feature-count", type=int, default=5)
        sp.add_argument("--weight-mode", choices=["none", "exp", "step"], default="step")
        sp.add_argument("--half-life-days", type=float, default=365.0)
        sp.add_argument("--recent-start", default="2023-01-01")
        sp.add_argument("--recent-weight", type=float, default=2.0)
        sp.add_argument("--market-csv", default="")
        sp.add_argument("--backtest-output", default="")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.cmd == "train-predict":
        rep = train_predict_weighted(train_jsonl=a.train_jsonl, eval_jsonl=a.eval_jsonl, predictions_output=a.predictions_output, alpha=a.alpha, threshold=a.threshold, min_gap=a.min_gap, target=a.target, min_feature_count=a.min_feature_count, weight_mode=a.weight_mode, half_life_days=a.half_life_days, recent_start=a.recent_start, recent_weight=a.recent_weight)
    else:
        rep = rolling_weighted(history_jsonl=a.history_jsonl, eval_jsonl=a.eval_jsonl, predictions_output=a.predictions_output, summary_output=a.summary_output, start_date=a.start_date, end_date=a.end_date, alpha=a.alpha, threshold=a.threshold, min_gap=a.min_gap, target=a.target, min_feature_count=a.min_feature_count, weight_mode=a.weight_mode, half_life_days=a.half_life_days, recent_start=a.recent_start, recent_weight=a.recent_weight)
    out: dict[str, Any] = {"summary": rep}
    if a.market_csv and a.backtest_output:
        bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=a.predictions_output, market_csv=a.market_csv, output=a.backtest_output))
        out["backtest"] = {"period": bt["period"], "sim": bt["sim"], "trade_stats": bt["trade_stats"]}
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
