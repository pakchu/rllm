"""Opportunity/event detection baseline for economic value rows.

This tests the hypothesis that the trading problem is dominated by rare
opportunity detection.  It learns whether a signal has an oracle best utility
above a threshold from past-only summary features, then backtests either an
oracle action upper bound or a simple train-prior action for detected events.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.economic_action_backtest import run_economic_action_backtest
from training.economic_value_baseline import FeatureSpace, _action_obj, load_jsonl, row_features, write_jsonl


def _signal_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1)))


def best_rows_by_signal(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_signal_key(row)].append(row)
    best: list[dict[str, Any]] = []
    for key in sorted(grouped):
        best.append(max(grouped[key], key=lambda r: float(r.get("utility", 0.0))))
    return best


def signal_feature_row(best_row: dict[str, Any]) -> dict[str, Any]:
    # Use prompt features only; neutral NO_TRADE action prevents leaking the oracle action into event detection.
    return {**best_row, "action": json.dumps({"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}, sort_keys=True)}


def fit_logistic(x: np.ndarray, y: np.ndarray, *, lr: float = 0.2, l2: float = 1.0, epochs: int = 400) -> np.ndarray:
    w = np.zeros(x.shape[1], dtype=np.float64)
    n = max(1.0, float(len(y)))
    for _ in range(int(epochs)):
        logits = np.clip(x @ w, -40, 40)
        p = 1.0 / (1.0 + np.exp(-logits))
        grad = x.T @ (p - y) / n + float(l2) * w / n
        w -= float(lr) * grad
    return w


def _metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = score >= float(threshold)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "predicted_positive": tp + fp, "actual_positive": tp + fn}


def _majority_profitable_action(train_best: list[dict[str, Any]], utility_threshold: float) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in train_best:
        if float(row.get("utility", 0.0)) >= float(utility_threshold):
            counts[str(row.get("action", "{}"))] += 1
    if not counts:
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
    return _action_obj(counts.most_common(1)[0][0])


def build_predictions(
    eval_best: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    score_threshold: float,
    mode: str,
    fallback_action: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, score in zip(eval_best, scores):
        if float(score) < float(score_threshold):
            action = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
        elif mode == "oracle_best":
            action = _action_obj(str(row.get("action", "{}")))
        elif mode == "majority_train_action":
            action = dict(fallback_action)
        else:
            raise ValueError("mode must be oracle_best or majority_train_action")
        out.append({"date": row.get("date"), "signal_pos": row.get("signal_pos"), "prediction": action, "event_score": float(score), "oracle_utility": float(row.get("utility", 0.0))})
    return out


def run_opportunity_baseline(
    *,
    train_jsonl: str,
    eval_jsonl: str,
    market_csv: str,
    output: str,
    predictions_output: str,
    utility_threshold: float = 0.005,
    score_threshold: float = 0.5,
    mode: str = "oracle_best",
    min_feature_count: int = 2,
    lr: float = 0.2,
    l2: float = 1.0,
    epochs: int = 400,
) -> dict[str, Any]:
    train_best = best_rows_by_signal(load_jsonl(train_jsonl))
    eval_best = best_rows_by_signal(load_jsonl(eval_jsonl))
    train_feat = [signal_feature_row(r) for r in train_best]
    eval_feat = [signal_feature_row(r) for r in eval_best]
    fs = FeatureSpace(min_count=min_feature_count)
    fs.fit(train_feat)
    x_train = fs.matrix(train_feat, fit_scale=True)
    x_eval = fs.matrix(eval_feat)
    y_train = np.array([1.0 if float(r.get("utility", 0.0)) >= float(utility_threshold) else 0.0 for r in train_best], dtype=np.float64)
    y_eval = np.array([1.0 if float(r.get("utility", 0.0)) >= float(utility_threshold) else 0.0 for r in eval_best], dtype=np.float64)
    w = fit_logistic(x_train, y_train, lr=lr, l2=l2, epochs=epochs)
    train_scores = 1.0 / (1.0 + np.exp(-np.clip(x_train @ w, -40, 40)))
    eval_scores = 1.0 / (1.0 + np.exp(-np.clip(x_eval @ w, -40, 40)))
    fallback_action = _majority_profitable_action(train_best, utility_threshold)
    preds = build_predictions(eval_best, eval_scores, score_threshold=score_threshold, mode=mode, fallback_action=fallback_action)
    write_jsonl(predictions_output, preds)
    bt_path = str(Path(output).with_suffix(".strict_backtest.json"))
    bt = run_economic_action_backtest(predictions_jsonl=predictions_output, market_csv=market_csv, output=bt_path)
    counts = Counter(f"{p['prediction']['gate']}/{p['prediction']['side']}/{p['prediction'].get('hold_bars',0)}" for p in preds)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "train_jsonl": train_jsonl,
        "eval_jsonl": eval_jsonl,
        "config": {"utility_threshold": utility_threshold, "score_threshold": score_threshold, "mode": mode, "features": len(fs.names), "lr": lr, "l2": l2, "epochs": epochs},
        "train_event_metrics": _metrics(y_train, train_scores, score_threshold),
        "eval_event_metrics": _metrics(y_eval, eval_scores, score_threshold),
        "train_positive_rate": float(y_train.mean()) if len(y_train) else 0.0,
        "eval_positive_rate": float(y_eval.mean()) if len(y_eval) else 0.0,
        "fallback_action": fallback_action,
        "prediction_counts": dict(sorted(counts.items())),
        "strict_backtest_path": bt_path,
        "strict_backtest": bt["backtest"],
        "leakage_guard": {"features_are_past_only": True, "eval_utility_not_used_for_fit_or_selection": True, "oracle_best_mode_is_upper_bound_not_live_policy": mode == "oracle_best"},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate economic opportunity detector")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--utility-threshold", type=float, default=0.005)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--mode", choices=["oracle_best", "majority_train_action"], default="oracle_best")
    p.add_argument("--min-feature-count", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=400)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_opportunity_baseline(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
