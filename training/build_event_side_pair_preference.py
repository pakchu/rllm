"""Build chosen/rejected preference pairs for event side-map side selection."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BuildEventSidePairPreferenceCfg:
    input_jsonl: str
    output_jsonl: str
    summary_output: str = ""


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _target_label(row: dict[str, Any]) -> str:
    obj = json.loads(str(row.get("target", "{}")))
    label = str(obj.get("side_pair", obj.get("side_map", ""))).strip().lower()
    if label not in {"normal", "inverse"}:
        return ""
    return label


def _response(label: str) -> str:
    return json.dumps({"side_pair": str(label)}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build(cfg: BuildEventSidePairPreferenceCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    out: list[dict[str, Any]] = []
    chosen_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for row in rows:
        chosen = _target_label(row)
        if not chosen:
            skipped["non_pair_label"] += 1
            continue
        rejected = "inverse" if chosen == "normal" else "normal"
        nr = {
            "task": "event_side_pair_preference",
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "month": row.get("month"),
            "generated_side": row.get("generated_side"),
            "prompt": str(row.get("prompt", "")),
            "chosen": _response(chosen),
            "rejected": _response(rejected),
            "chosen_side_pair": chosen,
            "rejected_side_pair": rejected,
            "leakage_guard": {
                "prompt_reused_from_causal_event_side_pair_record": True,
                "chosen_rejected_use_future_realized_side_returns_for_training_only": True,
                "unreliable_rows_removed_from_preference_training": True,
            },
        }
        out.append(nr)
        chosen_counts[chosen] += 1
    _write_jsonl(cfg.output_jsonl, out)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows_in": len(rows),
        "pairs_out": len(out),
        "chosen_counts": dict(chosen_counts),
        "skipped_counts": dict(skipped),
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build event side-pair preference rows")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(build(BuildEventSidePairPreferenceCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
