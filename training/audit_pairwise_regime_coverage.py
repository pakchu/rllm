"""Audit pairwise-ranking regime coverage between train and evaluation splits.

The pairwise LLM target is only meaningful when final holdout regimes were
represented in training or are explicitly abstained. This script checks bucket
coverage and optional prediction accuracy by bucket.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def parse_choice(text: Any) -> str:
    if isinstance(text, dict):
        return str(text.get("choice", "")).upper()
    try:
        return str(json.loads(str(text)).get("choice", "")).upper()
    except Exception:
        return str(text).upper()


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p:
        return []
    return load_jsonl(p)


def summarize(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    train_buckets = Counter(str(r.get("bucket", "NA")) for r in train_rows)
    eval_buckets = Counter(str(r.get("bucket", "NA")) for r in eval_rows)
    unseen = {b: n for b, n in eval_buckets.items() if train_buckets.get(b, 0) == 0}
    low_seen = {b: n for b, n in eval_buckets.items() if 0 < train_buckets.get(b, 0) < 30}
    report: dict[str, Any] = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_bucket_counts": dict(train_buckets.most_common()),
        "eval_bucket_counts": dict(eval_buckets.most_common()),
        "eval_unseen_bucket_rows": sum(unseen.values()),
        "eval_unseen_buckets": unseen,
        "eval_low_coverage_buckets_train_lt_30": low_seen,
    }
    if pred_rows:
        by_bucket: dict[str, list[Any]] = defaultdict(lambda: [0, 0, Counter(), Counter()])
        known_correct = known_total = 0
        unknown_total = 0
        for row, pred in zip(eval_rows, pred_rows):
            bucket = str(row.get("bucket", "NA"))
            target = parse_choice(pred.get("target", row.get("target")))
            prediction = parse_choice(pred.get("prediction", ""))
            ok = prediction == target
            by_bucket[bucket][0] += int(ok)
            by_bucket[bucket][1] += 1
            by_bucket[bucket][2][prediction] += 1
            by_bucket[bucket][3][target] += 1
            if train_buckets.get(bucket, 0) == 0:
                unknown_total += 1
            else:
                known_correct += int(ok)
                known_total += 1
        report["prediction_rows"] = len(pred_rows)
        report["accuracy_known_train_buckets_only"] = known_correct / known_total if known_total else None
        report["known_bucket_rows"] = known_total
        report["unknown_bucket_rows"] = unknown_total
        report["prediction_by_bucket"] = {
            b: {
                "accuracy": v[0] / v[1] if v[1] else None,
                "rows": v[1],
                "train_rows": train_buckets.get(b, 0),
                "predictions": dict(v[2]),
                "targets": dict(v[3]),
            }
            for b, v in sorted(by_bucket.items(), key=lambda kv: (-kv[1][1], kv[0]))
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_jsonl)
    eval_rows = load_jsonl(args.eval_jsonl)
    pred_rows = load_predictions(args.predictions_jsonl) if args.predictions_jsonl else None
    report = summarize(train_rows, eval_rows, pred_rows)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
