"""Filter JSONL rows by lexicographic timestamp/date field bounds."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FilterJsonlByDateCfg:
    input_jsonl: str
    output_jsonl: str
    date_field: str = "date"
    min_date: str = ""
    max_date: str = ""
    include_min: bool = True
    include_max: bool = False


def _keep(value: str, cfg: FilterJsonlByDateCfg) -> bool:
    if cfg.min_date:
        if value < cfg.min_date or (value == cfg.min_date and not cfg.include_min):
            return False
    if cfg.max_date:
        if value > cfg.max_date or (value == cfg.max_date and not cfg.include_max):
            return False
    return True


def run(cfg: FilterJsonlByDateCfg) -> dict[str, Any]:
    in_path = Path(cfg.input_jsonl)
    out_path = Path(cfg.output_jsonl)
    kept: list[dict[str, Any]] = []
    total = 0
    missing = 0
    for line in in_path.read_text().splitlines():
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        raw = row.get(cfg.date_field)
        if raw is None:
            missing += 1
            continue
        if _keep(str(raw), cfg):
            kept.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + ("\n" if kept else ""))
    return {
        "config": asdict(cfg),
        "input_rows": total,
        "kept_rows": len(kept),
        "dropped_rows": total - len(kept),
        "missing_date_rows": missing,
        "first_date": kept[0].get(cfg.date_field) if kept else None,
        "last_date": kept[-1].get(cfg.date_field) if kept else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter JSONL rows by date/timestamp string bounds")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--date-field", default=FilterJsonlByDateCfg.date_field)
    p.add_argument("--min-date", default="")
    p.add_argument("--max-date", default="")
    p.add_argument("--include-min", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-max", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(FilterJsonlByDateCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
