"""Build class-balanced decision-analyzer SFT training files.

The first decision SFT run collapsed to all-ABSTAIN because chronological train
contains many more ABSTAIN examples than TRADE/FADE.  This utility creates a
training-only balanced file by downsampling majority decisions and oversampling
minority decisions with replacement.  Validation/OOS files must stay untouched.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.split_edge_decay_sft import read_jsonl, write_jsonl


@dataclass(frozen=True)
class BalanceDecisionConfig:
    target_per_decision: int = 0
    seed: int = 42
    balance_key: str = "decision"


def _target(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("target", "{}")
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def balance_decision_rows(rows: list[dict[str, Any]], cfg: BalanceDecisionConfig) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = str(_target(row).get(cfg.balance_key, ""))
        buckets[label].append(row)
    if not buckets:
        return []
    target_n = int(cfg.target_per_decision) if int(cfg.target_per_decision) > 0 else max(len(v) for v in buckets.values())
    rng = random.Random(int(cfg.seed))
    selected: list[dict[str, Any]] = []
    for label in sorted(buckets):
        bucket = list(buckets[label])
        if len(bucket) >= target_n:
            chosen = rng.sample(bucket, target_n)
        else:
            chosen = list(bucket)
            chosen.extend(rng.choice(bucket) for _ in range(target_n - len(bucket)))
        for i, row in enumerate(chosen):
            out = dict(row)
            out["sampling"] = {
                "balanced_decision_train": True,
                "balance_key": cfg.balance_key,
                "source_count_for_label": len(bucket),
                "target_count_for_label": target_n,
                "sample_index_in_label": i,
                "seed": int(cfg.seed),
            }
            selected.append(out)
    selected.sort(key=lambda r: (str(_target(r).get(cfg.balance_key, "")), str(r.get("date", "")), int((r.get("sampling") or {}).get("sample_index_in_label", 0))))
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = {"decision": Counter(), "action_side": Counter(), "confidence": Counter(), "rationale_class": Counter()}
    for row in rows:
        target = _target(row)
        for key in counts:
            counts[key][str(target.get(key, ""))] += 1
    return {key: dict(counter) for key, counter in counts.items()}


def build_balanced_decision_sft(*, input_jsonl: str, output: str, summary_output: str = "", target_per_decision: int = 0, seed: int = 42, balance_key: str = "decision") -> dict[str, Any]:
    cfg = BalanceDecisionConfig(target_per_decision=int(target_per_decision), seed=int(seed), balance_key=str(balance_key))
    rows = read_jsonl(input_jsonl)
    balanced = balance_decision_rows(rows, cfg)
    write_jsonl(output, balanced)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(Path(input_jsonl).resolve()),
        "output": output,
        "config": asdict(cfg),
        "source_rows": len(rows),
        "balanced_rows": len(balanced),
        "source_counts": summarize(rows),
        "balanced_counts": summarize(balanced),
        "leakage_guard": {
            "training_only_resampling": True,
            "validation_oos_must_not_be_resampled": True,
            "prompts_are_unchanged_past_only": True,
            "targets_are_future_path_teacher_labels": True,
        },
    }
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Balance decision-analyzer SFT train data")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--target-per-decision", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--balance-key", default="decision")
    return p.parse_args()


def main() -> None:
    print(json.dumps(build_balanced_decision_sft(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
