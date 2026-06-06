"""Build compact pressure-analyzer SFT rows.

The verbose analyzer summary made Gemma miss labels that are learnable by a
cheap structured baseline.  This converter keeps only compact past-only fields
that overlap with the structured baseline features.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.economic_path_shape_sft_data import _summary_from_prompt, load_jsonl
from training.economic_preference_sft_data import write_jsonl
from training.economic_value_baseline import _summary_obj

TOP_LEVEL_FIELDS = (
    "regime",
    "trend_alignment",
    "trend_strength",
    "location",
    "momentum",
    "oscillator",
    "risk_state",
    "volatility_level",
    "volume_state",
    "candle_pattern",
)
SYMBOLIC_FIELDS = (
    "Macro Dollar State",
    "Korea Premium State",
    "Order Flow",
    "Location",
    "Oscillator",
    "Risk State",
    "Trend Alignment",
    "Volume State",
    "Candle Pattern",
)
EVIDENCE_FIELDS = (
    "momentum_1h_pct",
    "momentum_2h_pct",
    "momentum_8h_pct",
    "range_position",
    "volume_zscore",
    "window_drawdown_pct",
    "window_volatility_pct",
)
SEQUENCE_FIELDS = (
    "drop_or_down",
    "flat",
    "lower_rejections",
    "rally_or_up",
    "upper_rejections",
    "volume_active_or_surge",
    "wide_or_extreme",
)


def compact_summary_from_prompt(prompt: str) -> dict[str, Any]:
    s = _summary_obj(_summary_from_prompt(prompt))
    sym = s.get("symbolic_features", {}) if isinstance(s.get("symbolic_features"), dict) else {}
    ev = s.get("evidence", {}) if isinstance(s.get("evidence"), dict) else {}
    seq = s.get("sequence_stats", {}) if isinstance(s.get("sequence_stats"), dict) else {}
    out: dict[str, Any] = {}
    out["state"] = {k: s.get(k) for k in TOP_LEVEL_FIELDS if s.get(k) is not None}
    out["symbolic"] = {k: sym.get(k) for k in SYMBOLIC_FIELDS if sym.get(k) is not None}
    out["evidence"] = {k: ev.get(k) for k in EVIDENCE_FIELDS if ev.get(k) is not None}
    out["sequence"] = {k: seq.get(k) for k in SEQUENCE_FIELDS if seq.get(k) is not None}
    tags = s.get("context_tags", [])
    if isinstance(tags, list):
        out["tags"] = [str(t) for t in tags[:10]]
    return out


def compact_pressure_prompt(compact: dict[str, Any]) -> str:
    return (
        "Classify BTCUSDT futures short-horizon path pressure from compact past-only features.\n"
        "Return exactly one JSON object: {\"direction_pressure\": <LABEL>}\n"
        "Allowed LABEL values: LONG_FAVORED, SHORT_FAVORED, NO_TRADE_FAVORED, BOTH_SIDES_VOLATILE.\n"
        "Use side only when the path is expected to hit the target before the stop; otherwise prefer NO_TRADE_FAVORED.\n\n"
        f"Compact features: {json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
    )


def convert_compact_pressure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        target = row.get("analyzer_target", {}) if isinstance(row.get("analyzer_target"), dict) else {}
        pressure = str(target.get("direction_pressure", "NO_TRADE_FAVORED"))
        compact = compact_summary_from_prompt(str(row.get("prompt", "")))
        out.append(
            {
                "task": "compact_path_pressure_analyzer_sft",
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "prompt": compact_pressure_prompt(compact),
                "target": json.dumps({"direction_pressure": pressure}, ensure_ascii=False, sort_keys=True),
                "pressure": pressure,
                "compact_features": compact,
                "leakage_guard": {"prompt_uses_future_path": False, "target_uses_future_path_pressure_for_training_only": True},
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(r.get("pressure")) for r in rows)
    prompt_lens = [len(str(r.get("prompt", ""))) for r in rows]
    target_lens = [len(str(r.get("target", ""))) for r in rows]
    return {
        "rows": len(rows),
        "period": {"start": rows[0].get("date") if rows else None, "end": rows[-1].get("date") if rows else None},
        "pressure_counts": dict(sorted(counts.items())),
        "majority_baseline_accuracy": max(counts.values()) / max(1, len(rows)) if counts else 0.0,
        "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens) / max(1, len(prompt_lens))},
        "target_chars": {"min": min(target_lens) if target_lens else 0, "max": max(target_lens) if target_lens else 0, "mean": sum(target_lens) / max(1, len(target_lens))},
    }


def build_compact_pressure_sft(*, input_jsonl: str, output: str, summary_output: str = "") -> dict[str, Any]:
    rows = convert_compact_pressure_rows(load_jsonl(input_jsonl))
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
    print(json.dumps(build_compact_pressure_sft(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
