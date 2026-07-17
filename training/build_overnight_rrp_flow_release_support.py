"""Freeze outcome-blind ORFR-1 primary and mechanism-control clocks."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from training import overnight_rrp_flow_release_clock as clock
from training import preregister_overnight_rrp_flow_release as prereg


DEFAULT_OUTPUT = "results/overnight_rrp_flow_release_support_2026-07-17.json"
DEFAULT_CLOCKS = "results/overnight_rrp_flow_release_clocks_2026-07-17.csv.gz"
CLOCK_COLUMNS = (
    "control",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "operation_date",
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_registration() -> dict[str, Any]:
    manifest = json.loads(Path(prereg.DEFAULT_OUTPUT).read_text())
    prereg.validate_manifest(manifest)
    return manifest


def _event_rows(control: str, events: list[clock.Event]) -> list[dict[str, Any]]:
    return [
        {
            "control": control,
            "signal_time": event.decision_time,
            "entry_time": event.entry_time,
            "exit_time": event.scheduled_exit_time,
            "side": 1 if event.side == "LONG" else -1,
            "operation_date": event.operation_date,
        }
        for event in events
    ]


def _one_release_delay(
    primary: list[clock.Event], source_rows: list[clock.SourceRow]
) -> list[dict[str, Any]]:
    by_date = {row.operation_date: index for index, row in enumerate(source_rows)}
    delayed: list[dict[str, Any]] = []
    for event in primary:
        source_index = by_date[event.operation_date]
        if source_index + 2 >= len(source_rows):
            continue
        delayed_source = source_rows[source_index + 1]
        following_source = source_rows[source_index + 2]
        entry = delayed_source.available_at + timedelta(minutes=5)
        exit_time = following_source.available_at + timedelta(minutes=5)
        delayed.append(
            {
                "control": "one_release_delay",
                "signal_time": delayed_source.available_at.isoformat(),
                "entry_time": entry.isoformat(),
                "exit_time": exit_time.isoformat(),
                "side": 1 if event.side == "LONG" else -1,
                "operation_date": delayed_source.operation_date,
            }
        )
    if any(
        clock._parse_utc(left["exit_time"]) > clock._parse_utc(right["entry_time"])
        for left, right in zip(delayed, delayed[1:])
    ):
        raise ValueError("ORFR delayed control overlaps")
    return delayed


def build_clock_rows(rows: list[clock.SourceRow]) -> list[dict[str, Any]]:
    primary = clock.build_events(rows)
    ledger = _event_rows("primary", primary)
    ledger.extend(
        _event_rows(
            "one_day_delta_tail",
            clock.build_events(rows, mode="one_day_delta"),
        )
    )
    ledger.extend(_one_release_delay(primary, rows))
    return sorted(
        ledger,
        key=lambda row: (row["control"], row["entry_time"], row["operation_date"]),
    )


def _write_clocks(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=list(CLOCK_COLUMNS), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(cast(Any, rows))


def _window(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    return [row for row in rows if start <= row["entry_time"] < end]


def _summary(rows: list[dict[str, Any]], start: str, end: str) -> dict[str, Any]:
    selected = _window(rows, start, end)
    months: dict[str, int] = {}
    for row in selected:
        month = row["entry_time"][:7]
        months[month] = months.get(month, 0) + 1
    return {
        "events": len(selected),
        "longs": sum(row["side"] == 1 for row in selected),
        "shorts": sum(row["side"] == -1 for row in selected),
        "max_single_month_count": max(months.values(), default=0),
        "max_single_month_share": (
            max(months.values(), default=0) / len(selected) if selected else 0.0
        ),
    }


def _jaccard(
    left: list[dict[str, Any]], right: list[dict[str, Any]], start: str, end: str
) -> float:
    a = {row["entry_time"] for row in _window(left, start, end)}
    b = {row["entry_time"] for row in _window(right, start, end)}
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def build_report(
    *, clocks_path: str | Path = DEFAULT_CLOCKS, write_clock: bool = True
) -> dict[str, Any]:
    registration = _load_registration()
    source_rows = clock.read_source()
    ledger = build_clock_rows(source_rows)
    if write_clock:
        _write_clocks(clocks_path, ledger)
    grouped = {
        name: [row for row in ledger if row["control"] == name]
        for name in ("primary", "one_day_delta_tail", "one_release_delay")
    }
    windows = {
        "2021": ("2021-01-01T00:00:00+00:00", "2022-01-01T00:00:00+00:00"),
        "2022": ("2022-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"),
        "train": ("2021-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"),
        "2023_h1": ("2023-01-01T00:00:00+00:00", "2023-07-01T00:00:00+00:00"),
        "2023_h2": ("2023-07-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
        "2023": ("2023-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
    }
    summaries = {
        control: {
            name: _summary(rows, *window) for name, window in windows.items()
        }
        for control, rows in grouped.items()
    }
    primary = summaries["primary"]
    gates = registration["support_gates"]
    checks = {
        "train_events": primary["train"]["events"] >= gates["minimum_train_events"],
        "2021_events": primary["2021"]["events"] >= gates["minimum_each_stage1_year"],
        "2022_events": primary["2022"]["events"] >= gates["minimum_each_stage1_year"],
        "train_longs": primary["train"]["longs"] >= gates["minimum_train_each_side"],
        "train_shorts": primary["train"]["shorts"] >= gates["minimum_train_each_side"],
        "2023_events": primary["2023"]["events"] >= gates["minimum_2023_events"],
        "2023_h1_events": primary["2023_h1"]["events"] >= gates["minimum_each_2023_half"],
        "2023_h2_events": primary["2023_h2"]["events"] >= gates["minimum_each_2023_half"],
        "2023_longs": primary["2023"]["longs"] >= gates["minimum_2023_each_side"],
        "2023_shorts": primary["2023"]["shorts"] >= gates["minimum_2023_each_side"],
        "train_month_share": primary["train"]["max_single_month_share"]
        <= gates["maximum_single_month_share"],
        "2023_month_share": primary["2023"]["max_single_month_share"]
        <= gates["maximum_single_month_share"],
        "nine_quarantined_rows": sum(not row.source_complete for row in source_rows)
        == 9,
        "quarantined_values_hidden": all(
            row.amount_usd is not None if row.source_complete else row.amount_usd is None
            for row in source_rows
        ),
    }
    passed = all(checks.values())
    clock_file = Path(clocks_path)
    core = {
        "protocol_version": "overnight_rrp_flow_release_support_v1",
        "policy_id": "ORFR-1",
        "as_of_date": "2026-07-17",
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "market_or_funding_rows_opened": 0,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "policy": asdict(prereg.Policy()),
        "source_rows": len(source_rows),
        "source_complete_rows": sum(row.source_complete for row in source_rows),
        "source_quarantined_rows": sum(not row.source_complete for row in source_rows),
        "clock_summaries": summaries,
        "source_clock_jaccard": {
            control: {
                "train": _jaccard(grouped["primary"], rows, *windows["train"]),
                "2023": _jaccard(grouped["primary"], rows, *windows["2023"]),
            }
            for control, rows in grouped.items()
            if control != "primary"
        },
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_stage1_outcomes": passed,
        "clocks": {
            "path": str(clock_file),
            "sha256": _sha256(clock_file) if write_clock else None,
            "rows": len(ledger),
            "columns": list(CLOCK_COLUMNS),
        },
        "disposition": "FREEZE_EVALUATOR" if passed else "REJECT_BEFORE_OUTCOMES",
    }
    return {**core, "manifest_hash": _canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clocks", default=DEFAULT_CLOCKS)
    args = parser.parse_args()
    report = build_report(clocks_path=args.clocks)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
