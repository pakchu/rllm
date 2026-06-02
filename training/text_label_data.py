"""Build plain single-label SFT rows from JSON-key trader tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.eval_text_json_key import parse_key_json


def _extract_summary(prompt: str) -> str:
    marker = "Analyzer summary:"
    return prompt.split(marker, 1)[1].strip() if marker in prompt else prompt.strip()


def build_label_rows(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    key = str(key).strip().lower()
    if key not in {"gate", "side"}:
        raise ValueError("key must be one of {'gate','side'}")
    labels = "TRADE or NO_TRADE" if key == "gate" else "LONG or SHORT"
    out: list[dict[str, Any]] = []
    for row in rows:
        label = parse_key_json(str(row["target"]), key=key)
        common = {k: v for k, v in row.items() if k not in {"task", "prompt", "target"}}
        summary = _extract_summary(str(row["prompt"]))
        out.append(
            {
                **common,
                "task": f"plain_{key}",
                "prompt": "\n".join(
                    [
                        f"You are the {key} classifier for a BTCUSDT futures trading bot.",
                        "Use only the analyzer's past-only symbolic market summary.",
                        f"Output exactly one label: {labels}.",
                        "Do not output JSON, prose, punctuation, or extra tokens.",
                        "",
                        f"Analyzer summary: {summary}",
                    ]
                ),
                "target": label,
            }
        )
    return out


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def build_label_jsonl(*, input_jsonl: str, output_jsonl: str, key: str, summary_output: str = "") -> dict[str, Any]:
    rows = build_label_rows(read_jsonl(input_jsonl), key=key)
    write_jsonl(output_jsonl, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["target"])] = counts.get(str(row["target"]), 0) + 1
    summary = {
        "input_jsonl": str(Path(input_jsonl).resolve()),
        "output_jsonl": output_jsonl,
        "key": key,
        "records": len(rows),
        "target_counts": dict(sorted(counts.items())),
        "leakage_guard": {"prompts_use_analyzer_summary_only": True, "targets_are_plain_labels": True},
    }
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build plain-label SFT rows from JSON-key rows")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--key", choices=["gate", "side"], required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(build_label_jsonl(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
