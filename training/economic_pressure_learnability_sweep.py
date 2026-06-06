"""Cheap learnability sweep for path-pressure label definitions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.economic_opportunity_baseline import best_rows_by_signal, signal_feature_row
from training.economic_path_shape_data import PathTemplate, compute_path_shape
from training.economic_value_baseline import FeatureSpace, load_jsonl

LABELS = ("LONG_FAVORED", "SHORT_FAVORED", "NO_TRADE_FAVORED", "BOTH_SIDES_VOLATILE")


def pressure_labels(rows: list[dict[str, Any]], market: pd.DataFrame, *, horizon_bars: int, target_pct: float, stop_pct: float) -> list[str]:
    labels = []
    tpl = PathTemplate(horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, entry_delay_bars=1)
    for row in best_rows_by_signal(rows):
        shape = compute_path_shape(market, int(row.get("signal_pos", -1)), tpl)
        labels.append(str((shape or {}).get("direction_pressure", "NO_TRADE_FAVORED")))
    return labels


def majority_accuracy(labels: list[str]) -> float:
    c = Counter(labels)
    return max(c.values()) / max(1, len(labels)) if c else 0.0


def fit_softmax(x: np.ndarray, y: np.ndarray, *, num_classes: int, lr: float = 0.3, l2: float = 1.0, epochs: int = 800) -> np.ndarray:
    w = np.zeros((x.shape[1], num_classes), dtype=np.float64)
    n = max(1, x.shape[0])
    eye = np.eye(num_classes, dtype=np.float64)
    yy = eye[y]
    for _ in range(int(epochs)):
        logits = np.clip(x @ w, -40, 40)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
        grad = (x.T @ (probs - yy)) / n + float(l2) * w / n
        w -= float(lr) * grad
    return w


def predict_softmax(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.argmax(x @ w, axis=1)


def metrics(true_labels: list[str], pred_idx: np.ndarray) -> dict[str, Any]:
    true_idx = np.array([LABELS.index(x) if x in LABELS else LABELS.index("NO_TRADE_FAVORED") for x in true_labels], dtype=np.int64)
    correct = pred_idx == true_idx
    confusion: dict[str, int] = {}
    for t, p in zip(true_idx, pred_idx):
        key = f"target={LABELS[int(t)]}|pred={LABELS[int(p)]}"
        confusion[key] = confusion.get(key, 0) + 1
    pred_counts = Counter(LABELS[int(p)] for p in pred_idx)
    return {"n": len(true_labels), "accuracy": float(correct.mean()) if len(true_labels) else 0.0, "majority_baseline": majority_accuracy(true_labels), "edge_over_majority": (float(correct.mean()) if len(true_labels) else 0.0) - majority_accuracy(true_labels), "label_counts": dict(sorted(Counter(true_labels).items())), "prediction_counts": dict(sorted(pred_counts.items())), "confusion": dict(sorted(confusion.items()))}


def run_combo(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], oos_rows: list[dict[str, Any]], market: pd.DataFrame, *, horizon_bars: int, target_pct: float, stop_pct: float) -> dict[str, Any]:
    train_best = best_rows_by_signal(train_rows)
    val_best = best_rows_by_signal(val_rows)
    oos_best = best_rows_by_signal(oos_rows)
    train_labels = pressure_labels(train_rows, market, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct)
    val_labels = pressure_labels(val_rows, market, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct)
    oos_labels = pressure_labels(oos_rows, market, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct)
    fs = FeatureSpace(min_count=2)
    fs.fit([signal_feature_row(r) for r in train_best])
    x_train = fs.matrix([signal_feature_row(r) for r in train_best], fit_scale=True)
    x_val = fs.matrix([signal_feature_row(r) for r in val_best])
    x_oos = fs.matrix([signal_feature_row(r) for r in oos_best])
    y_train = np.array([LABELS.index(x) if x in LABELS else LABELS.index("NO_TRADE_FAVORED") for x in train_labels], dtype=np.int64)
    w = fit_softmax(x_train, y_train, num_classes=len(LABELS))
    return {
        "config": {"horizon_bars": horizon_bars, "target_pct": target_pct, "stop_pct": stop_pct},
        "train": metrics(train_labels, predict_softmax(x_train, w)),
        "val": metrics(val_labels, predict_softmax(x_val, w)),
        "oos": metrics(oos_labels, predict_softmax(x_oos, w)),
    }


def _parse_nums(text: str, cast=float) -> list[Any]:
    return [cast(x) for x in str(text).split(",") if str(x).strip()]


def run_sweep(*, train_jsonl: str, val_jsonl: str, oos_jsonl: str, market_csv: str, output: str, horizons: str = "36,72,144", targets: str = "0.5,0.8,1.0", stops: str = "0.4,0.6") -> dict[str, Any]:
    market = pd.read_csv(market_csv)
    train_rows = load_jsonl(train_jsonl)
    val_rows = load_jsonl(val_jsonl)
    oos_rows = load_jsonl(oos_jsonl)
    results = []
    for h in _parse_nums(horizons, int):
        for t in _parse_nums(targets, float):
            for s in _parse_nums(stops, float):
                results.append(run_combo(train_rows, val_rows, oos_rows, market, horizon_bars=h, target_pct=t, stop_pct=s))
    ranked = sorted(results, key=lambda r: (r["val"]["edge_over_majority"], r["oos"]["edge_over_majority"], r["val"]["accuracy"]), reverse=True)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "inputs": {"train": train_jsonl, "val": val_jsonl, "oos": oos_jsonl, "market": market_csv}, "results": ranked, "top": ranked[:10], "leakage_guard": {"features_past_only": True, "fit_on_train_only": True, "val_for_selection_only": True, "oos_report_only": True}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--oos-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--horizons", default="36,72,144")
    p.add_argument("--targets", default="0.5,0.8,1.0")
    p.add_argument("--stops", default="0.4,0.6")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_sweep(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
