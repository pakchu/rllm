"""Cheap KNN baseline for event-action policy learnability."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from training.economic_action_backtest import run_economic_action_backtest


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _state_from_prompt(prompt: str) -> dict[str, float]:
    m = re.search(r"Past-only state:\s*(\{.*\})", str(prompt))
    if not m:
        return {}
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return {}
    return {str(k): float(v) for k, v in obj.items()}


def _target(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(str(row["target"]))


def _fit_matrix(rows: list[dict[str, Any]], keys: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    states = [_state_from_prompt(str(r.get("prompt", ""))) for r in rows]
    if keys is None:
        keys = sorted({k for s in states for k in s})
    x = np.asarray([[float(s.get(k, 0.0)) for k in keys] for s in states], dtype=float)
    return x, keys


def _standardize(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-9] = 1.0
    return (train_x - mean) / std, (eval_x - mean) / std


def _vote(actions: list[dict[str, Any]], distances: np.ndarray) -> dict[str, Any]:
    # Weighted vote by exact JSON action; tie-break by nearest.
    weights = 1.0 / np.maximum(distances, 1e-6)
    scores: dict[str, float] = {}
    first: dict[str, dict[str, Any]] = {}
    for action, w in zip(actions, weights):
        key = json.dumps(action, sort_keys=True, separators=(",", ":"))
        scores[key] = scores.get(key, 0.0) + float(w)
        first.setdefault(key, action)
    best_key = max(scores, key=scores.get)
    return first[best_key]


def predict_knn(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], *, k: int) -> list[dict[str, Any]]:
    train_x, keys = _fit_matrix(train_rows)
    eval_x, _ = _fit_matrix(eval_rows, keys)
    train_z, eval_z = _standardize(train_x, eval_x)
    train_actions = [_target(r) for r in train_rows]
    preds: list[dict[str, Any]] = []
    kk = max(1, min(int(k), len(train_rows)))
    for row, vec in zip(eval_rows, eval_z):
        d = np.linalg.norm(train_z - vec, axis=1)
        idx = np.argsort(d)[:kk]
        action = _vote([train_actions[int(i)] for i in idx], d[idx])
        preds.append({"date": row["date"], "signal_pos": row["signal_pos"], "prediction": action})
    return preds


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _accuracy(eval_rows: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    exact = gate = side = hold = 0
    confusion = Counter()
    for row, pred in zip(eval_rows, preds):
        tgt = _target(row)
        p = pred["prediction"]
        exact += int(p == tgt)
        gate += int(p.get("gate") == tgt.get("gate"))
        side += int(p.get("side") == tgt.get("side"))
        hold += int(int(p.get("hold_bars", 0) or 0) == int(tgt.get("hold_bars", 0) or 0))
        confusion[f"t={tgt.get('gate')}/{tgt.get('side')}/{tgt.get('hold_bars')}|p={p.get('gate')}/{p.get('side')}/{p.get('hold_bars')}"] += 1
    n = max(1, len(eval_rows))
    return {
        "rows": len(eval_rows),
        "exact": exact / n,
        "gate": gate / n,
        "side": side / n,
        "hold": hold / n,
        "top_confusion": dict(confusion.most_common(20)),
    }


def run_knn_baseline(*, train_jsonl: str, eval_jsonl: str, market_csv: str, output_predictions: str, output_report: str, k: int = 25) -> dict[str, Any]:
    train_rows = _load_jsonl(train_jsonl)
    eval_rows = _load_jsonl(eval_jsonl)
    preds = predict_knn(train_rows, eval_rows, k=int(k))
    _write_jsonl(output_predictions, preds)
    bt = run_economic_action_backtest(
        predictions_jsonl=output_predictions,
        market_csv=market_csv,
        output=str(Path(output_report).with_suffix(".backtest.json")),
        leverage=0.5,
        max_hold_bars=432,
    )
    report = {
        "inputs": {"train_jsonl": train_jsonl, "eval_jsonl": eval_jsonl, "market_csv": market_csv},
        "k": int(k),
        "predictions": output_predictions,
        "accuracy": _accuracy(eval_rows, preds),
        "backtest": bt["backtest"],
        "leakage_guard": {
            "fit_uses_train_rows_only": True,
            "eval_targets_used_only_for_accuracy_audit": True,
            "backtest_uses_predictions_only": True,
        },
    }
    Path(output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(output_report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KNN baseline for event-action policy")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-predictions", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--k", type=int, default=25)
    return p.parse_args()


def main() -> None:
    out = run_knn_baseline(**vars(parse_args()))
    print(json.dumps({"k": out["k"], "accuracy": out["accuracy"], "backtest": out["backtest"]["sim"], "trade_stats": out["backtest"]["trade_stats"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
