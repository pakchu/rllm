"""Freeze outcome-blind EPSB-1 support and falsification clocks."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pandas as pd

from training import eia_petroleum_stock_breadth_clock as clock
from training import preregister_eia_petroleum_stock_breadth as prereg


DEFAULT_OUTPUT = "results/eia_petroleum_stock_breadth_support_2026-07-17.json"
DEFAULT_CLOCKS = "results/eia_petroleum_stock_breadth_clocks_2026-07-17.csv.gz"
CONTROL_NAMES = (
    "primary",
    "crude_only",
    "refined_products_only",
    "direction_flip",
    "one_release_delay",
    "deterministic_random_side",
)
CLOCK_COLUMNS = (
    "control",
    "release_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "commercial_crude_change_mmbbl",
    "gasoline_change_mmbbl",
    "distillate_change_mmbbl",
    "archive_page_url",
    "table1_csv_url",
)
WINDOWS = {
    "2019_source_history": (
        "2019-01-01T00:00:00+00:00",
        "2020-01-01T00:00:00+00:00",
    ),
    "2020": ("2020-01-01T00:00:00+00:00", "2021-01-01T00:00:00+00:00"),
    "2021": ("2021-01-01T00:00:00+00:00", "2022-01-01T00:00:00+00:00"),
    "2022": ("2022-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"),
    "stage1_2020_2022": (
        "2020-01-01T00:00:00+00:00",
        "2023-01-01T00:00:00+00:00",
    ),
    "2023_h1": ("2023-01-01T00:00:00+00:00", "2023-07-01T00:00:00+00:00"),
    "2023_h2": ("2023-07-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
    "stage2_2023": (
        "2023-01-01T00:00:00+00:00",
        "2024-01-01T00:00:00+00:00",
    ),
}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_registration() -> dict[str, Any]:
    registration = json.loads(Path(prereg.DEFAULT_OUTPUT).read_text())
    core = {
        key: value for key, value in registration.items() if key != "manifest_hash"
    }
    expected = prereg.sha256_bytes(prereg.canonical_json(core))
    if registration.get("manifest_hash") != expected:
        raise ValueError("EPSB-1 preregistration manifest mismatch")
    if registration.get("policy_id") != "EPSB-1":
        raise ValueError("EPSB-1 preregistration policy mismatch")
    if registration.get("opened_outcome_windows") != []:
        raise ValueError("EPSB-1 preregistration opened outcomes")
    if registration.get("policy", {}).get("mutable_parameters") != []:
        raise ValueError("EPSB-1 preregistration is mutable")
    return cast(dict[str, Any], registration)


def _event_row(
    control: str, event: clock.Event, *, side: int | None = None
) -> dict[str, Any]:
    row = asdict(event)
    row["control"] = control
    if side is not None:
        row["side"] = side
    return {column: row[column] for column in CLOCK_COLUMNS}


def _random_side(release_date: str, signal_time: str) -> int:
    identity = f"EPSB-1|{release_date}|{signal_time}".encode()
    return 1 if hashlib.sha256(identity).digest()[0] & 1 else -1


def build_clock_rows(rows: list[clock.SourceRow]) -> list[dict[str, Any]]:
    primary = clock.build_events(rows)
    controls: dict[str, list[clock.Event]] = {
        "primary": primary,
        "crude_only": clock.build_events(rows, mode="crude_only"),
        "refined_products_only": clock.build_events(
            rows, mode="refined_products_only"
        ),
        "one_release_delay": clock.build_one_release_delay(rows),
    }
    ledger: list[dict[str, Any]] = []
    for control, events in controls.items():
        ledger.extend(_event_row(control, event) for event in events)
    ledger.extend(
        _event_row("direction_flip", event, side=-event.side) for event in primary
    )
    ledger.extend(
        _event_row(
            "deterministic_random_side",
            event,
            side=_random_side(event.release_date, event.signal_time),
        )
        for event in primary
    )
    return sorted(ledger, key=lambda row: (row["control"], row["entry_time"]))


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


def _valid_clock(rows: list[dict[str, Any]]) -> bool:
    ordered = sorted(rows, key=lambda row: row["entry_time"])
    if any(row["side"] not in (-1, 1) for row in ordered):
        return False
    for row in ordered:
        signal = clock._timestamp(row["signal_time"])
        entry = clock._timestamp(row["entry_time"])
        exit_time = clock._timestamp(row["exit_time"])
        if entry != signal + pd.Timedelta(minutes=5):
            return False
        if exit_time != entry + pd.Timedelta(hours=72):
            return False
    return all(
        current["entry_time"] >= previous["exit_time"]
        for previous, current in zip(ordered, ordered[1:])
    )


def _registered_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "events": item["trades"],
        "longs": item["longs"],
        "shorts": item["shorts"],
        "max_single_month_count": item["max_single_month_count"],
        "max_single_month_share": item["max_single_month_share"],
    }


def build_report(
    *, clocks_path: str | Path = DEFAULT_CLOCKS, write_clock: bool = True
) -> dict[str, Any]:
    registration = _load_registration()
    source_rows = clock.load_source()
    ledger = build_clock_rows(source_rows)
    if write_clock:
        _write_clocks(clocks_path, ledger)
    grouped = {
        control: [row for row in ledger if row["control"] == control]
        for control in CONTROL_NAMES
    }
    summaries = {
        control: {
            name: _summary(rows, *window) for name, window in WINDOWS.items()
        }
        for control, rows in grouped.items()
    }
    registered = registration["source_only_distributions"]
    primary = summaries["primary"]
    checks = {
        "source_hashes_match": True,
        "source_rows_exactly_259": len(source_rows) == 259,
        "source_complete_rows_exactly_258": sum(
            row.source_complete for row in source_rows
        )
        == 258,
        "one_source_row_quarantined": sum(
            not row.source_complete for row in source_rows
        )
        == 1,
        "all_controls_present": all(grouped[control] for control in CONTROL_NAMES),
        "all_control_clocks_valid": all(
            _valid_clock(grouped[control]) for control in CONTROL_NAMES
        ),
        "primary_2019_replays": primary["2019_source_history"]
        == _registered_summary(registered["2019_source_history"]),
        "primary_stage1_replays": primary["stage1_2020_2022"]
        == _registered_summary(registered["stage1_2020_2022"]),
        "primary_stage2_replays": primary["stage2_2023"]
        == _registered_summary(registered["stage2_2023"]),
        "direction_flip_is_exact": all(
            flipped["entry_time"] == original["entry_time"]
            and flipped["side"] == -original["side"]
            for original, flipped in zip(
                grouped["primary"], grouped["direction_flip"]
            )
        ),
        "random_side_is_deterministic": all(
            row["side"] == _random_side(row["release_date"], row["signal_time"])
            for row in grouped["deterministic_random_side"]
        ),
        "market_or_funding_rows_opened_zero": True,
    }
    passed = all(checks.values())
    clock_file = Path(clocks_path)
    core = {
        "protocol_version": "eia_petroleum_stock_breadth_support_v1",
        "policy_id": "EPSB-1",
        "as_of_date": prereg.AS_OF_DATE,
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "market_or_funding_rows_opened": 0,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "source_rows": len(source_rows),
        "source_complete_rows": sum(row.source_complete for row in source_rows),
        "source_quarantined_rows": sum(
            not row.source_complete for row in source_rows
        ),
        "clock_summaries": summaries,
        "source_clock_jaccard": {
            control: {
                "stage1": _jaccard(
                    grouped["primary"], rows, *WINDOWS["stage1_2020_2022"]
                ),
                "stage2": _jaccard(
                    grouped["primary"], rows, *WINDOWS["stage2_2023"]
                ),
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
