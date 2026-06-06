"""Convert path-shape rows into analyzer/trader SFT datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.economic_preference_sft_data import write_jsonl


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _summary_from_prompt(prompt: str) -> str:
    marker = "Past-only analyzer summary: "
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt.strip()


def analyzer_prompt(summary: str) -> str:
    return (
        "You are a futures market path-shape analyzer. Use only the past-only analyzer summary.\n"
        "Predict the future path diagnostics as compact JSON with keys: direction_pressure, long_path, short_path.\n"
        "Do not output a trade order; describe target/stop risk shape only.\n\n"
        f"Past-only analyzer summary: {summary}"
    )


def trader_prompt(summary: str, analyzer_target: dict[str, Any]) -> str:
    analyzer_view = {
        "direction_pressure": analyzer_target.get("direction_pressure"),
        "long_path": analyzer_target.get("long_path"),
        "short_path": analyzer_target.get("short_path"),
        "template": analyzer_target.get("template"),
    }
    return (
        "You are a futures trader using a path-shape analyzer output.\n"
        "Choose exactly one JSON action from LONG, SHORT, or NO_TRADE using the provided stop/target template.\n"
        "Return keys: gate, side, target_pct, stop_pct, max_hold_bars.\n\n"
        f"Past-only analyzer summary: {summary}\n\n"
        f"Analyzer path-shape output: {json.dumps(analyzer_view, ensure_ascii=False, sort_keys=True)}"
    )


def trader_target(analyzer_target: dict[str, Any]) -> dict[str, Any]:
    template = analyzer_target.get("template", {}) if isinstance(analyzer_target.get("template"), dict) else {}
    pressure = analyzer_target.get("direction_pressure")
    if pressure == "LONG_FAVORED":
        return {"gate": "TRADE", "side": "LONG", "target_pct": template.get("target_pct", 1.0), "stop_pct": template.get("stop_pct", 0.6), "max_hold_bars": template.get("horizon_bars", 144)}
    if pressure == "SHORT_FAVORED":
        return {"gate": "TRADE", "side": "SHORT", "target_pct": template.get("target_pct", 1.0), "stop_pct": template.get("stop_pct", 0.6), "max_hold_bars": template.get("horizon_bars", 144)}
    return {"gate": "NO_TRADE", "side": "NONE", "target_pct": 0.0, "stop_pct": 0.0, "max_hold_bars": 0}


def convert_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    analyzer_rows = []
    trader_rows = []
    for row in rows:
        summary = _summary_from_prompt(str(row.get("prompt", "")))
        target = row["analyzer_target"]
        analyzer_rows.append({
            "task": "path_shape_analyzer_sft",
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "prompt": analyzer_prompt(summary),
            "target": json.dumps(target, ensure_ascii=False, sort_keys=True),
            "leakage_guard": {"prompt_uses_future_path": False, "target_uses_future_path_for_training_only": True},
        })
        ttarget = trader_target(target)
        trader_rows.append({
            "task": "path_shape_trader_sft",
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "prompt": trader_prompt(summary, target),
            "target": json.dumps(ttarget, ensure_ascii=False, sort_keys=True),
            "pressure": target.get("direction_pressure"),
            "leakage_guard": {"prompt_includes_analyzer_output": True, "target_uses_path_shape_pressure_for_training_only": True},
        })
    return analyzer_rows, trader_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = Counter(str(r.get("pressure") or json.loads(r["target"]).get("direction_pressure") if str(r.get("target", "")).startswith("{") else "NA") for r in rows)
    prompt_lens = [len(str(r.get("prompt", ""))) for r in rows]
    target_lens = [len(str(r.get("target", ""))) for r in rows]
    return {
        "rows": len(rows),
        "period": {"start": rows[0].get("date") if rows else None, "end": rows[-1].get("date") if rows else None},
        "target_or_pressure_counts": dict(sorted(targets.items())),
        "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens) / max(1, len(prompt_lens))},
        "target_chars": {"min": min(target_lens) if target_lens else 0, "max": max(target_lens) if target_lens else 0, "mean": sum(target_lens) / max(1, len(target_lens))},
    }


def build_path_shape_sft(*, input_jsonl: str, analyzer_output: str, trader_output: str, summary_output: str = "") -> dict[str, Any]:
    rows = load_jsonl(input_jsonl)
    analyzer_rows, trader_rows = convert_rows(rows)
    write_jsonl(analyzer_output, analyzer_rows)
    write_jsonl(trader_output, trader_rows)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "input": input_jsonl,
        "outputs": {"analyzer": analyzer_output, "trader": trader_output},
        "analyzer": summarize(analyzer_rows),
        "trader": summarize(trader_rows),
    }
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--analyzer-output", required=True)
    p.add_argument("--trader-output", required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(build_path_shape_sft(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
