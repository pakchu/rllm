"""Export single-policy predictions from cheap key-wise Naive Bayes models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.decision_feature_learnability import load_jsonl
from training.eval_single_policy import DEFAULT_POLICY, parse_policy_json
from training.router_state_feature_learnability import KeyNaiveBayesModel

POLICY_KEYS = ("regime", "edge_quality", "risk", "action", "exit_profile", "confidence")


def _normalise_policy(policy: dict[str, Any]) -> dict[str, str]:
    return parse_policy_json(json.dumps(policy, sort_keys=True))


def predict_rows(train_rows: list[dict[str, Any]], rows: list[dict[str, Any]], *, alpha: float = 1.0) -> list[dict[str, Any]]:
    models = {key: KeyNaiveBayesModel.fit(train_rows, key=key, alpha=alpha) for key in POLICY_KEYS}
    out: list[dict[str, Any]] = []
    for row in rows:
        pred = dict(DEFAULT_POLICY)
        for key, model in models.items():
            value = model.predict(row)
            if value:
                pred[key] = value
        pred = _normalise_policy(pred)
        rec = dict(row)
        rec["target"] = json.dumps(pred, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        rec["policy_prediction"] = pred
        rec["policy_target"] = parse_policy_json(str(row.get("target", "{}")))
        rec["prediction_model"] = "keywise_categorical_naive_bayes_dependency_free"
        out.append(rec)
    return out


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = load_jsonl(args.train_jsonl)
    eval_rows = load_jsonl(args.eval_jsonl)
    if args.max_records:
        eval_rows = eval_rows[: int(args.max_records)]
    pred_rows = predict_rows(train_rows, eval_rows, alpha=float(args.alpha))
    write_jsonl(args.output, pred_rows)
    counts: dict[str, dict[str, int]] = {k: {} for k in POLICY_KEYS}
    for row in pred_rows:
        pred = row["policy_prediction"]
        for key in POLICY_KEYS:
            value = str(pred.get(key, ""))
            counts[key][value] = counts[key].get(value, 0) + 1
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {"train_jsonl": args.train_jsonl, "eval_jsonl": args.eval_jsonl},
        "output": args.output,
        "rows": len(pred_rows),
        "prediction_counts": counts,
        "leakage_guard": {
            "models_fit_on_train_only": True,
            "eval_targets_not_used_for_prediction": True,
            "prompt_features_are_past_only": True,
        },
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict single-policy JSON via train-only Naive Bayes baselines")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-records", type=int, default=0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
