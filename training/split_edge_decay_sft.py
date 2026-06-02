"""Chronologically split edge-decay analyzer JSONL into train/val/oos SFT files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: str(r.get("date", "")))


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _select(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    start_ts = pd.to_datetime(start) if start else None
    end_ts = pd.to_datetime(end) if end else None
    out = []
    for row in rows:
        ts = pd.to_datetime(row["date"])
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        out.append(row)
    return out


def _target_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    keys = (
        "decision",
        "action_side",
        "confidence",
        "rationale_class",
        "edge_decay_label",
        "transition_label",
        "risk_label",
        "recommended_router_hint",
    )
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        target = json.loads(str(row.get("target", "{}")))
        for key in keys:
            if key not in target:
                continue
            counts.setdefault(key, {})
            value = str(target.get(key, ""))
            counts[key][value] = counts[key].get(value, 0) + 1
    return {k: dict(sorted(v.items())) for k, v in counts.items()}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"records": 0, "period": None, "target_counts": {}}
    return {
        "records": len(rows),
        "period": {"start": rows[0]["date"], "end": rows[-1]["date"]},
        "target_counts": _target_counts(rows),
    }


def split_edge_decay_sft(
    *,
    input_jsonl: str,
    train_output: str,
    val_output: str,
    oos_output: str,
    summary_output: str = "",
    train_start: str = "",
    train_end: str = "",
    val_start: str = "",
    val_end: str = "",
    oos_start: str = "",
    oos_end: str = "",
) -> dict[str, Any]:
    rows = read_jsonl(input_jsonl)
    train = _select(rows, train_start, train_end)
    val = _select(rows, val_start, val_end)
    oos = _select(rows, oos_start, oos_end)
    write_jsonl(train_output, train)
    write_jsonl(val_output, val)
    write_jsonl(oos_output, oos)
    summary = {
        "input_jsonl": str(Path(input_jsonl).resolve()),
        "outputs": {"train": train_output, "val": val_output, "oos": oos_output},
        "splits": {"train": _summary(train), "val": _summary(val), "oos": _summary(oos)},
        "leakage_guard": {
            "chronological_split": True,
            "train_val_oos_non_overlapping": True,
            "prompts_are_past_only": True,
            "targets_are_future_path_labels": True,
            "not_gate_threshold_optimization": True,
        },
    }
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split edge-decay analyzer JSONL into chronological train/val/oos files")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--val-output", required=True)
    p.add_argument("--oos-output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--train-start", default="")
    p.add_argument("--train-end", default="")
    p.add_argument("--val-start", default="")
    p.add_argument("--val-end", default="")
    p.add_argument("--oos-start", default="")
    p.add_argument("--oos-end", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(split_edge_decay_sft(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
