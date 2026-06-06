"""Build pressure-only analyzer SFT rows from path-shape data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.economic_path_shape_sft_data import _summary_from_prompt, load_jsonl
from training.economic_preference_sft_data import write_jsonl


def pressure_prompt(summary: str) -> str:
    return (
        "You are a futures path-pressure classifier. Use only the past-only analyzer summary.\n"
        "Return exactly one compact JSON object with key direction_pressure.\n"
        "Allowed values: LONG_FAVORED, SHORT_FAVORED, NO_TRADE_FAVORED, BOTH_SIDES_VOLATILE.\n\n"
        f"Past-only analyzer summary: {summary}"
    )


def convert_pressure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        target = row.get("analyzer_target", {}) if isinstance(row.get("analyzer_target"), dict) else {}
        pressure = str(target.get("direction_pressure", "NO_TRADE_FAVORED"))
        out.append({
            "task": "path_pressure_analyzer_sft",
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "prompt": pressure_prompt(_summary_from_prompt(str(row.get("prompt", "")))),
            "target": json.dumps({"direction_pressure": pressure}, ensure_ascii=False, sort_keys=True),
            "pressure": pressure,
            "leakage_guard": {"prompt_uses_future_path": False, "target_uses_future_path_pressure_for_training_only": True},
        })
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(r.get("pressure")) for r in rows)
    prompt_lens = [len(str(r.get("prompt", ""))) for r in rows]
    target_lens = [len(str(r.get("target", ""))) for r in rows]
    majority = max(counts.values()) / max(1, len(rows)) if counts else 0.0
    return {
        "rows": len(rows),
        "period": {"start": rows[0].get("date") if rows else None, "end": rows[-1].get("date") if rows else None},
        "pressure_counts": dict(sorted(counts.items())),
        "majority_baseline_accuracy": majority,
        "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens) / max(1, len(prompt_lens))},
        "target_chars": {"min": min(target_lens) if target_lens else 0, "max": max(target_lens) if target_lens else 0, "mean": sum(target_lens) / max(1, len(target_lens))},
    }


def build_pressure_sft(*, input_jsonl: str, output: str, summary_output: str = "") -> dict[str, Any]:
    rows = convert_pressure_rows(load_jsonl(input_jsonl))
    write_jsonl(output, rows)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "input": input_jsonl, "output": output, "summary": summarize(rows)}
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(build_pressure_sft(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
