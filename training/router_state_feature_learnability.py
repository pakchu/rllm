"""Key-wise categorical baseline for repaired router-state targets."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from training.decision_feature_learnability import load_jsonl, record_features

DEFAULT_KEYS = (
    "trend_continuation_quality",
    "fade_warning",
    "skip_reason",
    "primary_route",
    "horizon_policy",
)


def parse_target(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("target", "{}")
    if isinstance(raw, dict):
        return dict(raw)
    try:
        obj = json.loads(str(raw))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


@dataclass
class KeyNaiveBayesModel:
    key: str
    labels: tuple[str, ...]
    label_counts: Counter[str]
    feature_value_counts: dict[str, dict[str, Counter[str]]]
    feature_values: dict[str, set[str]]
    alpha: float = 1.0

    @classmethod
    def fit(cls, rows: Iterable[dict[str, Any]], *, key: str, alpha: float = 1.0) -> "KeyNaiveBayesModel":
        label_counts: Counter[str] = Counter()
        feature_value_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        feature_values: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            label = str(parse_target(row).get(key, ""))
            if not label:
                continue
            label_counts[label] += 1
            for name, value in record_features(row).items():
                value = str(value)
                feature_value_counts[name][label][value] += 1
                feature_values[name].add(value)
        return cls(key=key, labels=tuple(sorted(label_counts)), label_counts=label_counts, feature_value_counts=dict(feature_value_counts), feature_values=dict(feature_values), alpha=alpha)

    def predict(self, row: dict[str, Any]) -> str:
        total = sum(self.label_counts.values())
        if total <= 0 or not self.labels:
            return ""
        feats = record_features(row)
        best_label = self.labels[0]
        best_score = float("-inf")
        for label in self.labels:
            prior = (self.label_counts.get(label, 0) + self.alpha) / (total + self.alpha * len(self.labels))
            score = math.log(prior)
            label_total = self.label_counts.get(label, 0)
            for name, value in feats.items():
                values = self.feature_values.get(name) or {str(value)}
                denom = label_total + self.alpha * (len(values) + 1)
                count = self.feature_value_counts.get(name, {}).get(label, Counter()).get(str(value), 0)
                score += math.log((count + self.alpha) / denom)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label


def evaluate_key(model: KeyNaiveBayesModel, rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets: Counter[str] = Counter()
    preds: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    correct = 0
    for row in rows:
        target = str(parse_target(row).get(model.key, ""))
        pred = model.predict(row)
        targets[target] += 1
        preds[pred] += 1
        confusion[(target, pred)] += 1
        correct += int(target == pred)
    n = len(rows)
    majority = max(targets.values()) / n if n and targets else 0.0
    acc = correct / n if n else 0.0
    return {
        "n": n,
        "accuracy": acc,
        "majority_baseline_accuracy": majority,
        "beats_majority_baseline": acc > majority if n else False,
        "target_counts": dict(targets),
        "prediction_counts": dict(preds),
        "confusion": {f"{t}->{p}": c for (t, p), c in sorted(confusion.items())},
    }


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    splits = {"train": load_jsonl(args.train_jsonl), "val": load_jsonl(args.val_jsonl)}
    if args.oos_jsonl:
        splits["oos"] = load_jsonl(args.oos_jsonl)
    keys = tuple(k.strip() for k in args.keys.split(",") if k.strip())
    per_key = {}
    for key in keys:
        model = KeyNaiveBayesModel.fit(splits["train"], key=key, alpha=float(args.alpha))
        per_key[key] = {
            "train_label_counts": dict(model.label_counts),
            "feature_count": len(model.feature_values),
            "splits": {name: evaluate_key(model, rows) for name, rows in splits.items()},
        }
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "model": "keywise_categorical_naive_bayes_dependency_free",
        "inputs": {"train_jsonl": args.train_jsonl, "val_jsonl": args.val_jsonl, "oos_jsonl": args.oos_jsonl},
        "alpha": float(args.alpha),
        "keys": list(keys),
        "per_key": per_key,
        "interpretation": {
            "go_no_go_rule": "Train LLM only for keys that beat majority on val and OOS or are economically justified by strict backtest.",
            "prompt_uses_future_path": False,
            "labels_use_future_path_teacher": True,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Key-wise repaired router-state feature learnability baseline")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--oos-jsonl", default="")
    p.add_argument("--output", default="results/router_state_feature_learnability.json")
    p.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    p.add_argument("--alpha", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_report(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
