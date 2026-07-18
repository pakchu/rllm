"""Freeze outcome-blind CFCS-1 support and control clocks."""

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

from training import cleveland_fed_cpi_surprise_clock as clock
from training import preregister_cleveland_fed_cpi_surprise as prereg


PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "9c252a988885c7fa1975b6f7190af4efeab50ee8541a67c0bb8f8882a3fa3e0d"
)
PRIMARY_CLOCK = Path(prereg.DEFAULT_CLOCK)
PRIMARY_CLOCK_SHA256 = (
    "cff8d0f8d7810400bc78f833cc91996a7b2cd0e9d5903fe0ef154f0e38a71739"
)
DEFAULT_OUTPUT = "results/cleveland_fed_cpi_surprise_support_2026-07-18.json"
DEFAULT_CLOCKS = "results/cleveland_fed_cpi_surprise_clocks_2026-07-18.csv.gz"
DEFAULT_DOCS = "docs/cleveland-fed-cpi-surprise-support-2026-07-18.md"
CONTROL_NAMES = (
    "primary",
    "headline_only",
    "core_only",
    "composite_no_concordance",
    "direction_flip",
    "one_day_delay",
    "seven_day_placebo",
)
CLOCK_COLUMNS = ("control", *clock.EVENT_COLUMNS)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_registration() -> dict[str, Any]:
    if _sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("CFCS-1 preregistration file changed")
    registration = json.loads(PREREGISTRATION.read_text())
    prereg.validate_manifest(registration)
    if registration.get("outcomes_opened") is not False:
        raise ValueError("CFCS-1 preregistration opened outcomes")
    if registration.get("opened_outcome_windows") != []:
        raise ValueError("CFCS-1 preregistration opened an outcome window")
    return cast(dict[str, Any], registration)


def _event_row(
    control: str, event: clock.Event, *, side: int | None = None
) -> dict[str, Any]:
    row = asdict(event)
    row["control"] = control
    if side is not None:
        row["side"] = side
    return {column: row[column] for column in CLOCK_COLUMNS}


def build_clock_rows(rows: list[clock.SourceRow]) -> list[dict[str, Any]]:
    primary = clock.build_events(rows)
    controls: dict[str, list[clock.Event]] = {
        "primary": primary,
        "headline_only": clock.build_events(rows, mode="headline_only"),
        "core_only": clock.build_events(rows, mode="core_only"),
        "composite_no_concordance": clock.build_events(
            rows, mode="composite_no_concordance"
        ),
        "one_day_delay": clock.build_events(rows, delay_days=1),
        "seven_day_placebo": clock.build_events(rows, delay_days=7),
    }
    ledger: list[dict[str, Any]] = []
    for control, events in controls.items():
        ledger.extend(_event_row(control, event) for event in events)
    ledger.extend(
        _event_row("direction_flip", event, side=-event.side) for event in primary
    )
    return sorted(ledger, key=lambda row: (row["control"], row["entry_time"]))


def _clock_bytes(rows: list[dict[str, Any]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(CLOCK_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(cast(Any, rows))
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.getvalue().encode())
    return output.getvalue()


def _window(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    return [row for row in rows if start <= row["entry_time"] < end]


def _summary(rows: list[dict[str, Any]], start: str, end: str) -> dict[str, Any]:
    selected = _window(rows, start, end)
    months: dict[str, int] = {}
    for row in selected:
        month = row["entry_time"][:7]
        months[month] = months.get(month, 0) + 1
    maximum = max(months.values(), default=0)
    return {
        "events": len(selected),
        "longs": sum(row["side"] == 1 for row in selected),
        "shorts": sum(row["side"] == -1 for row in selected),
        "months": len(months),
        "max_single_month_count": maximum,
        "max_single_month_share": maximum / len(selected) if selected else 0.0,
    }


def _jaccard(
    left: list[dict[str, Any]], right: list[dict[str, Any]], start: str, end: str
) -> float:
    a = {row["entry_time"] for row in _window(left, start, end)}
    b = {row["entry_time"] for row in _window(right, start, end)}
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _valid_clock(name: str, rows: list[dict[str, Any]]) -> bool:
    delay = 1 if name == "one_day_delay" else 7 if name == "seven_day_placebo" else 0
    ordered = sorted(rows, key=lambda row: row["entry_time"])
    if any(row["side"] not in (-1, 1) for row in ordered):
        return False
    for row in ordered:
        signal = clock._timestamp(row["signal_time"])
        expected_entry, expected_exit = clock._execution_times(signal, delay)
        if row["entry_time"] != expected_entry.isoformat():
            return False
        if row["exit_time"] != expected_exit.isoformat():
            return False
    return all(
        current["entry_time"] >= previous["exit_time"]
        for previous, current in zip(ordered, ordered[1:])
    )


def _verify_primary_clock(primary: list[clock.Event]) -> None:
    if _sha256(PRIMARY_CLOCK) != PRIMARY_CLOCK_SHA256:
        raise ValueError("CFCS-1 preregistered primary clock changed")
    frame = clock.events_frame(primary)
    expected = frame.to_csv(index=False, lineterminator="\n").encode()
    with gzip.open(PRIMARY_CLOCK, "rb") as handle:
        actual = handle.read()
    if actual != expected:
        raise ValueError("CFCS-1 primary clock no longer replays")


def build_report(
    *, clocks_path: str | Path = DEFAULT_CLOCKS, write_clock: bool = True
) -> dict[str, Any]:
    registration = _load_registration()
    source_rows = clock.load_source()
    _verify_primary_clock(clock.build_events(source_rows))
    ledger = build_clock_rows(source_rows)
    payload = _clock_bytes(ledger)
    if write_clock:
        prereg.write_once(clocks_path, payload)
    grouped = {
        control: [row for row in ledger if row["control"] == control]
        for control in CONTROL_NAMES
    }
    summaries = {
        control: {
            name: _summary(rows, start, end)
            for name, (start, end) in prereg.WINDOWS.items()
        }
        for control, rows in grouped.items()
    }
    registered = registration["research_history_boundary"]["disclosure"][
        "selected_clocks"
    ]["primary"]
    primary = summaries["primary"]
    support = registration["support_freeze_before_returns"]
    checks = {
        "source_hashes_match": True,
        "source_rows_exactly_60": len(source_rows) == 60,
        "all_controls_present": all(grouped[control] for control in CONTROL_NAMES),
        "all_control_clocks_valid": all(
            _valid_clock(control, grouped[control]) for control in CONTROL_NAMES
        ),
        "primary_clock_replays": True,
        "primary_source_counts_replay": all(
            primary[name] == registered[name] for name in prereg.WINDOWS
        ),
        "stage1_events": primary["stage1"]["events"] >= support["stage1_events_min"],
        "each_stage1_year": min(
            primary[year]["events"] for year in ("2020", "2021", "2022")
        )
        >= support["each_stage1_year_min"],
        "stage1_direction_support": (
            primary["stage1"]["longs"] >= support["stage1_long_min"]
            and primary["stage1"]["shorts"] >= support["stage1_short_min"]
        ),
        "sealed_2023_events": primary["2023"]["events"]
        >= support["sealed_2023_events_min"],
        "each_sealed_2023_half": min(
            primary[half]["events"] for half in ("2023_h1", "2023_h2")
        )
        >= support["each_sealed_2023_half_min"],
        "month_concentration": max(
            primary["stage1"]["max_single_month_share"],
            primary["2023"]["max_single_month_share"],
        )
        <= support["maximum_single_month_share"],
        "direction_flip_is_exact": all(
            flipped["entry_time"] == original["entry_time"]
            and flipped["exit_time"] == original["exit_time"]
            and flipped["side"] == -original["side"]
            for original, flipped in zip(grouped["primary"], grouped["direction_flip"])
        ),
        "market_or_funding_rows_opened_zero": True,
    }
    passed = all(checks.values())
    core: dict[str, Any] = {
        "protocol_version": "cleveland_fed_cpi_surprise_support_v1",
        "policy_id": "CFCS-1",
        "as_of_date": prereg.AS_OF_DATE,
        "preregistration_commit": "5d1901b",
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "market_or_funding_rows_opened": 0,
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "source": {
            "panel": str(clock.SOURCE),
            "panel_sha256": clock.SOURCE_SHA256,
            "build_manifest": str(clock.BUILD_MANIFEST),
            "build_manifest_sha256": clock.BUILD_MANIFEST_SHA256,
            "columns_loaded": list(clock.source_builder.PANEL_COLUMNS),
            "source_rows": len(source_rows),
        },
        "clock_summaries": summaries,
        "entry_clock_jaccard_vs_primary": {
            control: {
                "stage1": _jaccard(grouped["primary"], rows, *prereg.WINDOWS["stage1"]),
                "stage2": _jaccard(grouped["primary"], rows, *prereg.WINDOWS["2023"]),
            }
            for control, rows in grouped.items()
            if control != "primary"
        },
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_stage1_outcomes": passed,
        "clocks": {
            "path": str(clocks_path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": len(ledger),
            "counts": {name: len(rows) for name, rows in grouped.items()},
            "columns": list(CLOCK_COLUMNS),
        },
        "forbidden_opened": {
            "BTC_OHLC": False,
            "funding": False,
            "returns": False,
            "existing_alpha_overlap": False,
        },
        "disposition": "FREEZE_EVALUATOR" if passed else "REJECT_BEFORE_OUTCOMES",
    }
    return {**core, "manifest_hash": _canonical_hash(core)}


def validate_report(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("CFCS-1 support manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("CFCS-1 support opened outcomes")
    if payload.get("outcome_sources_opened") != []:
        raise ValueError("CFCS-1 support opened an outcome source")
    if payload.get("market_or_funding_rows_opened") != 0:
        raise ValueError("CFCS-1 support opened execution rows")
    if payload.get("support_passed") is not True:
        raise ValueError("CFCS-1 source support failed")


def render_docs(report: dict[str, Any]) -> str:
    primary = report["clock_summaries"]["primary"]
    rows = "\n".join(
        f"| {name} | {item['events']} | {item['longs']} | {item['shorts']} |"
        for name, item in primary.items()
    )
    checks = "\n".join(
        f"| `{name}` | {'PASS' if value else 'FAIL'} |"
        for name, value in report["support_checks"].items()
    )
    return f"""# CFCS-1 source-support freeze — 2026-07-18

## Decision

**{report["disposition"]}**

No BTC OHLC, funding, return, or existing-alpha overlap source was opened.
The exact primary and control clocks are now immutable.

## Primary source distribution

| Window | Events | Long | Short |
|---|---:|---:|---:|
{rows}

## Checks

| Check | Result |
|---|:---:|
{checks}

## Frozen controls

- headline-only, core-only, and no-concordance mechanism controls;
- exact direction flip;
- one-calendar-day delay and seven-calendar-day placebo;
- all entries use the same 08:35–16:00 America/New_York wall-clock window.

## Identity

- preregistration SHA-256: `{report["preregistration_sha256"]}`
- clock ledger SHA-256: `{report["clocks"]["sha256"]}`
- support manifest: `{report["manifest_hash"]}`
- market/funding rows opened: `{report["market_or_funding_rows_opened"]}`
"""


def run(
    *,
    output: str | Path = DEFAULT_OUTPUT,
    clocks_path: str | Path = DEFAULT_CLOCKS,
    docs: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    report = build_report(clocks_path=clocks_path)
    validate_report(report)
    prereg.write_once(output, prereg.canonical_json(report))
    prereg.write_once(docs, render_docs(report).encode())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clocks", default=DEFAULT_CLOCKS)
    parser.add_argument("--docs", default=DEFAULT_DOCS)
    args = parser.parse_args()
    report = run(output=args.output, clocks_path=args.clocks, docs=args.docs)
    print(
        json.dumps(
            {
                "policy_id": report["policy_id"],
                "support_passed": report["support_passed"],
                "outcomes_opened": False,
                "manifest_hash": report["manifest_hash"],
                "clock_sha256": report["clocks"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
