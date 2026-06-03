"""Check whether decision-analyzer prompt features are learnable before GPU SFT.

This is a cheap, dependency-free baseline for the current LLM supervision.  It
parses the same past-only analyzer summary that the decision LLM receives,
flattens symbolic fields and binned numeric evidence, then trains a categorical
Naive Bayes classifier on the train split.  If this baseline cannot beat a
majority-class rule on temporal validation/OOS splits, more LLM fine-tuning is
unlikely to fix the target without feature or label redesign.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from training.edge_decay_analyzer_data import write_jsonl

SUMMARY_MARKERS = (
    "Past-only context: ",
    "Past-only analyzer summary: ",
    "Analyzer summary: ",
)
DECISIONS = ("ABSTAIN", "TRADE_TREND", "FADE_TREND")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_jsonish(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
        return dict(value) if isinstance(value, dict) else {}
    except Exception:
        pass
    for marker in SUMMARY_MARKERS:
        if marker in text:
            candidate = text.rsplit(marker, 1)[-1].strip()
            try:
                value = json.loads(candidate)
                return dict(value) if isinstance(value, dict) else {}
            except Exception:
                # Fall through to brace extraction; old prompts sometimes wrap
                # natural language around the final JSON summary.
                text = candidate
                break
    match = re.search(r"\{.*\}\s*$", text, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            return {}
    return {}


def _num_bucket(value: float) -> str:
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        return "nan"
    av = abs(v)
    if av < 1e-9:
        return "zero"
    if v < 0:
        sign = "neg"
    else:
        sign = "pos"
    if av < 0.25:
        mag = "xs"
    elif av < 0.75:
        mag = "s"
    elif av < 1.5:
        mag = "m"
    elif av < 3.0:
        mag = "l"
    else:
        mag = "xl"
    return f"{sign}_{mag}"


def flatten_summary_features(summary: dict[str, Any]) -> dict[str, str]:
    """Flatten analyzer summary into categorical tokens for robust baselines."""
    out: dict[str, str] = {}

    def add(prefix: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            out[prefix] = str(value).lower()
        elif isinstance(value, (int, float)):
            out[prefix] = _num_bucket(float(value))
        elif isinstance(value, str):
            out[prefix] = value
        elif isinstance(value, list):
            out[prefix] = "|".join(str(x) for x in value)
            for item in value:
                out[f"{prefix}__has__{item}"] = "1"
        elif isinstance(value, dict):
            for k, v in value.items():
                add(f"{prefix}.{k}", v)
        else:
            out[prefix] = str(value)

    for key, value in summary.items():
        # numeric_feature_names is metadata, not a signal value.
        if key == "numeric_feature_names":
            continue
        add(key, value)
    return out


def record_features(record: dict[str, Any]) -> dict[str, str]:
    summary = parse_jsonish(record.get("past_summary")) or parse_jsonish(record.get("prompt"))
    return flatten_summary_features(summary)


def record_label(record: dict[str, Any]) -> str:
    target = parse_jsonish(record.get("target"))
    decision = str(target.get("decision", "ABSTAIN"))
    return decision if decision in DECISIONS else "ABSTAIN"


@dataclass
class NaiveBayesModel:
    label_counts: Counter[str]
    feature_value_counts: dict[str, dict[str, Counter[str]]]
    feature_values: dict[str, set[str]]
    labels: tuple[str, ...] = DECISIONS
    alpha: float = 1.0

    @classmethod
    def fit(cls, rows: Iterable[dict[str, Any]], *, alpha: float = 1.0) -> "NaiveBayesModel":
        label_counts: Counter[str] = Counter()
        feature_value_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        feature_values: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            label = record_label(row)
            label_counts[label] += 1
            feats = record_features(row)
            for name, value in feats.items():
                value = str(value)
                feature_value_counts[name][label][value] += 1
                feature_values[name].add(value)
        return cls(label_counts, dict(feature_value_counts), dict(feature_values), alpha=alpha)

    def predict(self, row: dict[str, Any]) -> str:
        total = sum(self.label_counts.values())
        if total <= 0:
            return "ABSTAIN"
        feats = record_features(row)
        best_label = "ABSTAIN"
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


def evaluate(model: NaiveBayesModel, rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: Counter[tuple[str, str]] = Counter()
    targets: Counter[str] = Counter()
    preds: Counter[str] = Counter()
    correct = 0
    for row in rows:
        target = record_label(row)
        pred = model.predict(row)
        targets[target] += 1
        preds[pred] += 1
        confusion[(target, pred)] += 1
        correct += int(target == pred)
    n = len(rows)
    majority = max(targets.values()) / n if n and targets else 0.0
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "majority_baseline_accuracy": majority,
        "beats_majority_baseline": (correct / n) > majority if n else False,
        "target_counts": dict(targets),
        "prediction_counts": dict(preds),
        "confusion": {f"{t}->{p}": c for (t, p), c in sorted(confusion.items())},
    }


def _opposite(side: str) -> str:
    if side == "LONG":
        return "SHORT"
    if side == "SHORT":
        return "LONG"
    return "NONE"


def _past_trend_side(row: dict[str, Any]) -> str:
    source_edge = parse_jsonish(row.get("source_edge_target"))
    side = str(source_edge.get("trend_side", "NONE"))
    return side if side in {"LONG", "SHORT"} else "NONE"


def prediction_record(row: dict[str, Any], decision: str) -> dict[str, Any]:
    trend_side = _past_trend_side(row)
    if decision == "TRADE_TREND":
        action_side = trend_side
    elif decision == "FADE_TREND":
        action_side = _opposite(trend_side)
    else:
        action_side = "NONE"
    if action_side not in {"LONG", "SHORT"}:
        decision = "ABSTAIN"
        action_side = "NONE"
    pred = {
        "decision": decision,
        "action_side": action_side,
        "confidence": "LOW",
        "rationale_class": "FEATURE_NB_BASELINE",
    }
    return {
        "date": row.get("date"),
        "signal_pos": row.get("signal_pos"),
        "prediction": json.dumps(pred, sort_keys=True, separators=(",", ":")),
        "target": row.get("target"),
        "task": "decision_feature_nb_baseline",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dependency-free decision feature learnability baseline")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--oos-jsonl", default="")
    p.add_argument("--output", default="results/decision_feature_learnability.json")
    p.add_argument("--prediction-output-dir", default="")
    p.add_argument("--alpha", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splits = {
        "train": load_jsonl(args.train_jsonl),
        "val": load_jsonl(args.val_jsonl),
    }
    if args.oos_jsonl:
        splits["oos"] = load_jsonl(args.oos_jsonl)
    model = NaiveBayesModel.fit(splits["train"], alpha=float(args.alpha))
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "model": "categorical_naive_bayes_dependency_free",
        "alpha": float(args.alpha),
        "inputs": {
            "train_jsonl": args.train_jsonl,
            "val_jsonl": args.val_jsonl,
            "oos_jsonl": args.oos_jsonl,
        },
        "feature_count": len(model.feature_values),
        "train_label_counts": dict(model.label_counts),
        "splits": {name: evaluate(model, rows) for name, rows in splits.items()},
        "interpretation": {
            "gpu_sft_go_no_go_rule": "If val/oos accuracy does not beat majority baseline, redesign features/targets before more LLM SFT.",
            "prompt_uses_future_path": False,
            "labels_use_future_path_teacher": True,
            "prediction_side_source": "past source_edge_target.trend_side; never target.action_side",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if args.prediction_output_dir:
        pred_root = Path(args.prediction_output_dir)
        pred_root.mkdir(parents=True, exist_ok=True)
        for name, rows in splits.items():
            preds = [prediction_record(row, model.predict(row)) for row in rows]
            write_jsonl(pred_root / f"{name}_predictions.jsonl", preds)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
