"""Convert REX event reasoning rows to equal-form multiple-choice labels.

JSON action completions caused candidate-logprob length/format bias.  This
builder emits canonical labels with near-identical format:
- CHOICE_A_LONG
- CHOICE_B_SHORT
- CHOICE_C_SKIP
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LABELS = {
    "LONG": "CHOICE_A_LONG",
    "SHORT": "CHOICE_B_SHORT",
    "NO_TRADE": "CHOICE_C_SKIP",
}
ACTIONS = {v: k for k, v in LABELS.items()}


@dataclass(frozen=True)
class Cfg:
    input_jsonl: str
    output_jsonl: str
    summary_output: str = ""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def parse_action(row: dict[str, Any]) -> str:
    obj = json.loads(str(row.get("target", "{}")))
    action = str(obj.get("action", "NO_TRADE")).upper()
    return action if action in LABELS else "NO_TRADE"


def convert_prompt(prompt: str) -> str:
    return "\n".join([
        str(prompt),
        "",
        "Multiple-choice output contract:",
        "- CHOICE_A_LONG means take LONG.",
        "- CHOICE_B_SHORT means take SHORT.",
        "- CHOICE_C_SKIP means no trade.",
        "Return exactly one label and nothing else.",
    ])


def build(cfg: Cfg) -> dict[str, Any]:
    rows = read_jsonl(cfg.input_jsonl)
    out: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = {"all": Counter(), "train": Counter(), "test": Counter(), "eval": Counter()}
    for row in rows:
        action = parse_action(row)
        label = LABELS[action]
        split = str(row.get("split", "unknown"))
        nr = dict(row)
        nr["task"] = "rex_event_choice_label_sft"
        nr["prompt"] = convert_prompt(str(row.get("prompt", "")))
        nr["target"] = label
        nr["target_action"] = action
        nr["choice_label"] = label
        nr["leakage_guard"] = dict(row.get("leakage_guard", {})) | {
            "choice_label_is_target_only": True,
            "prompt_contains_no_future_label": True,
            "equal_form_label_completion": True,
        }
        out.append(nr)
        counts.setdefault(split, Counter())[label] += 1
        counts["all"][label] += 1
    write_jsonl(cfg.output_jsonl, out)
    summary = {
        "config": asdict(cfg),
        "rows": len(out),
        "label_counts": {k: dict(v) for k, v in counts.items()},
        "prompt_chars": {
            "min": min((len(r["prompt"]) for r in out), default=0),
            "max": max((len(r["prompt"]) for r in out), default=0),
            "mean": sum(len(r["prompt"]) for r in out) / max(1, len(out)),
        },
        "target_chars": {"values": sorted(set(r["target"] for r in out))},
        "leakage_guard": {"source_prompts_are_signal_time_only": True, "future_path_used_only_for_target_label": True},
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    print(json.dumps(build(Cfg(**vars(p.parse_args()))), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
