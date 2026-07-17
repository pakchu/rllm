"""Freeze outcome-blind TADI-1 support and control clocks."""
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

from training import preregister_treasury_auction_demand_impulse as prereg
from training import treasury_auction_demand_impulse_clock as clock


DEFAULT_OUTPUT = "results/treasury_auction_demand_impulse_support_2026-07-17.json"
DEFAULT_CLOCKS = "results/treasury_auction_demand_impulse_clocks_2026-07-17.csv.gz"
CLOCK_COLUMNS = (
    "control",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "term",
    "cusip",
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


def _next_same_tenor_delay(
    events: list[clock.Event], rows: list[clock.SourceRow]
) -> list[dict[str, Any]]:
    complete_by_term: dict[str, list[clock.SourceRow]] = {}
    for row in rows:
        if row.source_complete:
            complete_by_term.setdefault(row.term, []).append(row)
    candidates: list[dict[str, Any]] = []
    for event in events:
        decision = clock._parse_utc(event.decision_time)
        later = [
            row
            for row in complete_by_term[event.original_security_term]
            if row.available_at > decision
        ]
        if not later:
            continue
        delayed = later[0]
        entry = delayed.available_at + timedelta(minutes=5)
        candidates.append(
            {
                "control": "one_auction_delay",
                "signal_time": delayed.available_at.isoformat(),
                "entry_time": entry.isoformat(),
                "exit_time": (entry + timedelta(hours=24)).isoformat(),
                "side": 1 if event.side == "LONG" else -1,
                "term": delayed.term,
                "cusip": delayed.cusip,
            }
        )
    reserved: list[dict[str, Any]] = []
    reserved_until = None
    for candidate in sorted(
        candidates,
        key=lambda row: (
            row["entry_time"],
            clock.TERM_PRIORITY[row["term"]],
            row["cusip"],
        ),
    ):
        entry = clock._parse_utc(candidate["entry_time"])
        exit_time = clock._parse_utc(candidate["exit_time"])
        if reserved_until is not None and entry < reserved_until:
            continue
        reserved.append(candidate)
        reserved_until = exit_time
    return reserved


def _event_rows(control: str, events: list[clock.Event]) -> list[dict[str, Any]]:
    return [
        {
            "control": control,
            "signal_time": event.decision_time,
            "entry_time": event.entry_time,
            "exit_time": event.scheduled_exit_time,
            "side": 1 if event.side == "LONG" else -1,
            "term": event.original_security_term,
            "cusip": event.cusip,
        }
        for event in events
    ]


def build_clock_rows(rows: list[clock.SourceRow]) -> list[dict[str, Any]]:
    primary = clock.build_events(rows)
    ledger = _event_rows("primary", primary)
    ledger.extend(
        _event_rows(
            "bid_to_cover_only",
            clock.build_events(rows, mode="bid_to_cover_only"),
        )
    )
    ledger.extend(
        _event_rows(
            "indirect_only", clock.build_events(rows, mode="indirect_only")
        )
    )
    ledger.extend(_next_same_tenor_delay(primary, rows))
    return sorted(
        ledger,
        key=lambda row: (
            row["control"],
            row["entry_time"],
            clock.TERM_PRIORITY[row["term"]],
        ),
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
    terms: dict[str, int] = {}
    for row in selected:
        months[row["entry_time"][:7]] = months.get(row["entry_time"][:7], 0) + 1
        terms[row["term"]] = terms.get(row["term"], 0) + 1
    return {
        "events": len(selected),
        "longs": sum(row["side"] == 1 for row in selected),
        "shorts": sum(row["side"] == -1 for row in selected),
        "term_counts": dict(sorted(terms.items())),
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
        for name in ("primary", "bid_to_cover_only", "indirect_only", "one_auction_delay")
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
        "2021_events": primary["2021"]["events"] >= gates["minimum_2021_events"],
        "2022_events": primary["2022"]["events"] >= gates["minimum_2022_events"],
        "2023_events": primary["2023"]["events"] >= gates["minimum_2023_events"],
        "2023_h1_events": primary["2023_h1"]["events"] >= gates["minimum_each_2023_half"],
        "2023_h2_events": primary["2023_h2"]["events"] >= gates["minimum_each_2023_half"],
        "train_longs": primary["train"]["longs"] >= gates["minimum_train_each_side"],
        "train_shorts": primary["train"]["shorts"] >= gates["minimum_train_each_side"],
        "2023_longs": primary["2023"]["longs"] >= gates["minimum_2023_each_side"],
        "2023_shorts": primary["2023"]["shorts"] >= gates["minimum_2023_each_side"],
        "train_month_share": primary["train"]["max_single_month_share"] <= gates["maximum_single_month_share"],
        "2023_month_share": primary["2023"]["max_single_month_share"] <= gates["maximum_single_month_share"],
        "five_quarantined_source_rows": sum(not row.source_complete for row in source_rows) == 5,
        "source_complete_rows_never_expose_none": all(
            (row.bid_to_cover is not None and row.indirect_share is not None)
            if row.source_complete
            else (row.bid_to_cover is None and row.indirect_share is None)
            for row in source_rows
        ),
    }
    passed = all(checks.values())
    clock_file = Path(clocks_path)
    core = {
        "protocol_version": "treasury_auction_demand_impulse_support_v1",
        "policy_id": "TADI-1",
        "as_of_date": "2026-07-17",
        "outcomes_opened": False,
        "outcome_sources_opened": [],
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
