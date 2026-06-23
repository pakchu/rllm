"""Project JSONL SFT targets to a single JSON key.

This keeps prompts/source metadata unchanged but replaces each target JSON with
`{key: value}`.  It is useful when candidate-logprob evaluation scores only one
key and the full training target contains extra fields that create train/eval
surface mismatch.
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
class ProjectJsonKeyTargetCfg:
    input_jsonl: str
    output_jsonl: str
    key: str
    summary_output: str = ""


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _project_target(raw: Any, key: str) -> dict[str, Any]:
    obj = json.loads(str(raw)) if not isinstance(raw, dict) else raw
    if key not in obj:
        raise KeyError(f"target missing key {key!r}: {obj}")
    return {key: obj[key]}


def project(cfg: ProjectJsonKeyTargetCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    out = []
    counts: Counter[str] = Counter()
    for row in rows:
        target_obj = _project_target(row.get("target", "{}"), cfg.key)
        nr = dict(row)
        nr["target"] = json.dumps(target_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        guard = dict(nr.get("leakage_guard", {}) if isinstance(nr.get("leakage_guard"), dict) else {})
        guard["target_projected_to_single_key"] = cfg.key
        nr["leakage_guard"] = guard
        counts[str(target_obj[cfg.key])] += 1
        out.append(nr)
    _write_jsonl(cfg.output_jsonl, out)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "rows": len(out), "target_counts": dict(counts)}
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project SFT target JSON to one key")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(project(ProjectJsonKeyTargetCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
