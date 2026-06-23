"""Build pairwise NORMAL-vs-INVERSE event side-map SFT rows.

The event side-map reliability label mixes two decisions in one 3-way target:
side direction transform (normal/inverse) and abstention (unreliable).  This
builder intentionally removes unreliable rows and projects the target to
`{"side_pair":"normal|inverse"}` so candidate-logprob does not let the abstain
label dominate side selection.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BuildEventSidePairCfg:
    input_jsonl: str
    output_jsonl: str
    summary_output: str = ""


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _side_map(row: dict[str, Any]) -> str:
    obj = json.loads(str(row.get("target", "{}")))
    return str(obj.get("side_map", "")).strip().lower()


def build(cfg: BuildEventSidePairCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for row in rows:
        label = _side_map(row)
        if label not in {"normal", "inverse"}:
            skipped[label or "missing"] += 1
            continue
        nr = dict(row)
        nr["target"] = json.dumps({"side_pair": label}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        nr["task"] = "event_side_pair_sft"
        guard = dict(nr.get("leakage_guard", {}) if isinstance(nr.get("leakage_guard"), dict) else {})
        guard["target_projected_to_pairwise_side_map"] = True
        guard["unreliable_rows_removed_from_side_pair_training"] = True
        nr["leakage_guard"] = guard
        counts[label] += 1
        out.append(nr)
    _write_jsonl(cfg.output_jsonl, out)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows_in": len(rows),
        "rows_out": len(out),
        "target_counts": dict(counts),
        "skipped_counts": dict(skipped),
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pairwise event side-map SFT rows")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(build(BuildEventSidePairCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
