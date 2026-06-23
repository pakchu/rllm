"""Diagnose policy false positives by causal prompt buckets."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.eval_dxy_kimchi_policy import parse_dxy_kimchi_policy


@dataclass(frozen=True)
class PolicyFalsePositiveDiagnosticsCfg:
    eval_jsonl: str
    predictions_jsonl: str
    output: str
    min_count: int = 5


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1) or -1))


def _extract_prompt_features(prompt: str) -> dict[str, str]:
    features: dict[str, str] = {}
    for raw in str(prompt).splitlines():
        line = raw.strip()
        if line.startswith("- ") and ":" in line:
            k, v = line[2:].split(":", 1)
            features[k.strip()] = v.strip()
    return features


def _decision(policy: dict[str, Any]) -> str:
    p = parse_dxy_kimchi_policy(json.dumps(policy, ensure_ascii=False))
    if bool(p.get("activate")) and p.get("action") in {"LONG", "SHORT"}:
        return str(p["action"])
    return "NO_TRADE"


def _category(target: str, pred: str) -> str:
    if target == pred:
        return "true_positive" if pred in {"LONG", "SHORT"} else "true_negative"
    if pred in {"LONG", "SHORT"} and target == "NO_TRADE":
        return f"false_positive_{pred}"
    if pred == "NO_TRADE" and target in {"LONG", "SHORT"}:
        return f"false_negative_{target}"
    return f"wrong_side_{target}_to_{pred}"


def run(cfg: PolicyFalsePositiveDiagnosticsCfg) -> dict[str, Any]:
    eval_rows = {_key(r): r for r in _load_jsonl(cfg.eval_jsonl)}
    pred_rows = _load_jsonl(cfg.predictions_jsonl)
    joined: list[dict[str, Any]] = []
    for prow in pred_rows:
        row = eval_rows.get(_key(prow))
        if row is None:
            continue
        target_policy = parse_dxy_kimchi_policy(str(row.get("target", "{}")))
        pred_policy = dict(prow.get("policy_prediction", {}))
        target = _decision(target_policy)
        pred = _decision(pred_policy)
        joined.append({
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "target": target,
            "pred": pred,
            "category": _category(target, pred),
            "features": _extract_prompt_features(str(row.get("prompt", ""))),
        })
    category_counts = Counter(r["category"] for r in joined)
    pred_counts = Counter(r["pred"] for r in joined)
    target_counts = Counter(r["target"] for r in joined)
    feature_stats: dict[str, dict[str, Any]] = {}
    feature_values: dict[str, Counter] = defaultdict(Counter)
    feature_value_cats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in joined:
        for k, v in row["features"].items():
            feature_values[k][v] += 1
            feature_value_cats[(k, v)][row["category"]] += 1
    for k, counts in sorted(feature_values.items()):
        rows = []
        for v, n in counts.most_common():
            if n < int(cfg.min_count):
                continue
            cats = feature_value_cats[(k, v)]
            fp = cats.get("false_positive_LONG", 0) + cats.get("false_positive_SHORT", 0)
            tp = cats.get("true_positive", 0)
            rows.append({
                "value": v,
                "n": n,
                "false_positive": fp,
                "true_positive": tp,
                "false_positive_rate": fp / max(1, n),
                "true_positive_rate": tp / max(1, n),
                "categories": dict(sorted(cats.items())),
            })
        if rows:
            feature_stats[k] = {"values": rows[:20]}
    report = {
        "config": asdict(cfg),
        "joined_rows": len(joined),
        "target_counts": dict(sorted(target_counts.items())),
        "prediction_counts": dict(sorted(pred_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "feature_stats": feature_stats,
        "leakage_guard": {"uses_prompt_features_only": True, "targets_used_for_diagnostics_only": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose policy false positives by causal prompt buckets")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-count", type=int, default=PolicyFalsePositiveDiagnosticsCfg.min_count)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(PolicyFalsePositiveDiagnosticsCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
