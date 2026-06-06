"""Drift/stability diagnostics for economic action datasets.

This is not a trader.  It reports whether labels, action utilities, and
train-memory action choices are stable across train/val/OOS without using eval
labels to select parameters.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import numpy as np

from training.economic_event_memory_policy import _action_obj, _action_text, build_action_memory, regime_key
from training.economic_opportunity_baseline import best_rows_by_signal, fit_logistic, signal_feature_row, _metrics
from training.economic_value_baseline import FeatureSpace, load_jsonl


def split_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return (row.get("date"), row.get("signal_pos"))


def utility_summary(values: Iterable[float]) -> dict[str, Any]:
    xs = [float(x) for x in values]
    if not xs:
        return {"n": 0}
    mu = mean(xs)
    sd = pstdev(xs) if len(xs) > 1 else 0.0
    se = sd / (len(xs) ** 0.5) if xs else 0.0
    return {
        "n": len(xs),
        "mean_pct": mu * 100.0,
        "std_pct": sd * 100.0,
        "win_rate": sum(x > 0 for x in xs) / len(xs),
        "ci95_mean_pct": [(mu - 1.96 * se) * 100.0, (mu + 1.96 * se) * 100.0],
    }


def rows_by_signal_action(rows: list[dict[str, Any]]) -> dict[tuple[Any, Any], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[Any, Any], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[split_key(row)][_action_text(_action_obj(str(row.get("action", "{}"))))] = row
    return grouped


def action_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(_action_text(_action_obj(str(r.get("action", "{}")))) for r in rows).items()))


def action_utility_table(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    vals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        vals[_action_text(_action_obj(str(row.get("action", "{}"))))].append(float(row.get("utility", 0.0)))
    return {k: utility_summary(v) for k, v in sorted(vals.items())}


def best_label_report(rows: list[dict[str, Any]], *, utility_threshold: float) -> dict[str, Any]:
    best = best_rows_by_signal(rows)
    positives = [r for r in best if float(r.get("utility", 0.0)) >= utility_threshold]
    return {
        "signals": len(best),
        "positive_signals": len(positives),
        "positive_rate": len(positives) / len(best) if best else 0.0,
        "best_utility": utility_summary(float(r.get("utility", 0.0)) for r in best),
        "positive_best_action_counts": action_distribution(positives),
        "all_best_action_counts": action_distribution(best),
    }


def fit_event_detector(train_best: list[dict[str, Any]], utility_threshold: float) -> tuple[FeatureSpace, np.ndarray]:
    fs = FeatureSpace(min_count=2)
    train_signal = [signal_feature_row(r) for r in train_best]
    fs.fit(train_signal)
    x_train = fs.matrix(train_signal, fit_scale=True)
    y_train = np.array([float(r.get("utility", 0.0)) >= utility_threshold for r in train_best], dtype=np.float64)
    w = fit_logistic(x_train, y_train, lr=0.2, l2=1.0, epochs=500)
    return fs, w


def score_event_detector(fs: FeatureSpace, w: np.ndarray, best: list[dict[str, Any]]) -> np.ndarray:
    x = fs.matrix([signal_feature_row(r) for r in best])
    return 1.0 / (1.0 + np.exp(-np.clip(x @ w, -40, 40)))


def memory_choice_report(
    *,
    train_best: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    split_best: list[dict[str, Any]],
    utility_threshold: float,
    fields: tuple[str, ...],
    min_bucket: int,
    event_scores: np.ndarray | None = None,
    event_threshold: float | None = None,
) -> dict[str, Any]:
    memory, fallback = build_action_memory(train_best, utility_threshold=utility_threshold, fields=fields, min_bucket=min_bucket)
    by_sig_action = rows_by_signal_action(split_rows)
    chosen_utils: list[float] = []
    missing = 0
    action_counts: Counter[str] = Counter()
    fired = 0
    for idx, best_row in enumerate(split_best):
        if event_scores is not None and event_threshold is not None and float(event_scores[idx]) < event_threshold:
            continue
        fired += 1
        action = memory.get(regime_key(best_row, fields), fallback)
        action_text = _action_text(action)
        action_counts[action_text] += 1
        row = by_sig_action.get(split_key(best_row), {}).get(action_text)
        if row is None:
            missing += 1
            continue
        chosen_utils.append(float(row.get("utility", 0.0)))
    return {
        "fields": list(fields),
        "min_bucket": min_bucket,
        "event_threshold": event_threshold,
        "memory_buckets": len(memory),
        "fallback": fallback,
        "signals_considered": fired if event_scores is not None else len(split_best),
        "missing_action_rows": missing,
        "chosen_action_counts": dict(sorted(action_counts.items())),
        "chosen_action_utility": utility_summary(chosen_utils),
    }


def run_diagnostic(
    *,
    train_jsonl: str,
    val_jsonl: str,
    oos_jsonl: str,
    output: str,
    utility_threshold: float = 0.003,
    fields: str = "trend_alignment,location,momentum",
    min_bucket: int = 3,
    event_threshold: float = 0.7,
) -> dict[str, Any]:
    splits = {
        "train": load_jsonl(train_jsonl),
        "val": load_jsonl(val_jsonl),
        "oos": load_jsonl(oos_jsonl),
    }
    best = {name: best_rows_by_signal(rows) for name, rows in splits.items()}
    fs, w = fit_event_detector(best["train"], utility_threshold)
    field_tuple = tuple(x.strip() for x in fields.split(",") if x.strip())
    report: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": {"utility_threshold": utility_threshold, "fields": list(field_tuple), "min_bucket": min_bucket, "event_threshold": event_threshold},
        "splits": {},
        "leakage_guard": {
            "event_detector_fit_on_train_only": True,
            "memory_fit_on_train_positive_events_only": True,
            "diagnostic_uses_eval_labels_for_reporting_only": True,
        },
    }
    for name, rows in splits.items():
        scores = score_event_detector(fs, w, best[name])
        y = np.array([float(r.get("utility", 0.0)) >= utility_threshold for r in best[name]], dtype=np.float64)
        report["splits"][name] = {
            "best_labels": best_label_report(rows, utility_threshold=utility_threshold),
            "candidate_action_utility": action_utility_table(rows),
            "event_metrics": _metrics(y, scores, event_threshold),
            "train_memory_all_signals": memory_choice_report(train_best=best["train"], split_rows=rows, split_best=best[name], utility_threshold=utility_threshold, fields=field_tuple, min_bucket=min_bucket),
            "train_memory_event_filtered": memory_choice_report(train_best=best["train"], split_rows=rows, split_best=best[name], utility_threshold=utility_threshold, fields=field_tuple, min_bucket=min_bucket, event_scores=scores, event_threshold=event_threshold),
        }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--oos-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--utility-threshold", type=float, default=0.003)
    p.add_argument("--fields", default="trend_alignment,location,momentum")
    p.add_argument("--min-bucket", type=int, default=3)
    p.add_argument("--event-threshold", type=float, default=0.7)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_diagnostic(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
