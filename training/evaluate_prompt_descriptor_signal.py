"""Cheap descriptor-signal audit for exported prompt samples."""
from __future__ import annotations

import argparse, json, math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DescriptorSignalConfig:
    samples_jsonl: str
    output: str
    fields: str = "Kimchi Flow Regime,Long Entry Context,Short Entry Context,Regime Failure Cue,Trade Readiness,Step Focus,Regime Memory,Regime Trap Risk"
    target_key: str = "target_action"


def _entropy(counts: Counter[str]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def _load_rows(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _field_report(rows: list[dict[str, Any]], field: str, target_key: str) -> dict[str, Any]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        target = str(row.get(target_key, "MISSING"))
        val = str(row.get("descriptors", {}).get(field, "MISSING"))
        groups[val][target] += 1
    base = Counter(str(row.get(target_key, "MISSING")) for row in rows)
    base_h = _entropy(base)
    cond_h = sum((sum(c.values()) / max(1, len(rows))) * _entropy(c) for c in groups.values())
    values = []
    for value, counts in sorted(groups.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(counts.values())
        majority_label, majority_n = counts.most_common(1)[0]
        values.append(
            {
                "value": value,
                "n": int(n),
                "counts": dict(sorted(counts.items())),
                "majority_label": majority_label,
                "majority_rate": majority_n / max(1, n),
            }
        )
    return {
        "field": field,
        "mutual_information_bits": base_h - cond_h,
        "values": values,
    }


def run(cfg: DescriptorSignalConfig) -> dict[str, Any]:
    rows = _load_rows(cfg.samples_jsonl)
    fields = [x.strip() for x in cfg.fields.split(",") if x.strip()]
    base = Counter(str(row.get(cfg.target_key, "MISSING")) for row in rows)
    field_reports = [_field_report(rows, field, cfg.target_key) for field in fields]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows": len(rows),
        "base_counts": dict(sorted(base.items())),
        "base_entropy_bits": _entropy(base),
        "fields_by_mi": sorted(field_reports, key=lambda x: x["mutual_information_bits"], reverse=True),
        "interpretation": {
            "low_mi_warning": "If all MI values are near zero, these descriptors are not useful for this target label; use a more aligned target such as regime activation or realized trade-quality gating.",
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit descriptor/target signal in exported prompt samples")
    p.add_argument("--samples-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fields", default=DescriptorSignalConfig.fields)
    p.add_argument("--target-key", default="target_action")
    return p.parse_args()


def main() -> None:
    report = run(DescriptorSignalConfig(**vars(parse_args())))
    print("rows", report["rows"], "base", report["base_counts"])
    for field in report["fields_by_mi"]:
        print(field["field"], "MI", round(field["mutual_information_bits"], 6), "top", field["values"][:4])


if __name__ == "__main__":
    main()
