"""Freeze outcome-blind H8DM-1 support and control clocks."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, cast

import pandas as pd

from training import fed_h8_deposit_migration_clock as clock
from training import preregister_fed_h8_deposit_migration as prereg


PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "0705042e6fceb5e183e5967be846bc106ea860f642fd44cba72dfa214eb09432"
)
PRIMARY_CLOCK = Path(prereg.DEFAULT_CLOCK)
PRIMARY_CLOCK_SHA256 = (
    "20405f79b86861adcc784c81223baae1c40fdf3c73edda339578471a6a6d1b40"
)
DEFAULT_OUTPUT = "results/fed_h8_deposit_migration_support_2026-07-18.json"
DEFAULT_CLOCKS = "results/fed_h8_deposit_migration_clocks_2026-07-18.csv.gz"
DEFAULT_DOCS = "docs/fed-h8-deposit-migration-support-2026-07-18.md"
CONTROL_NAMES = (
    "primary",
    "migration_only",
    "borrowings_only",
    "cash_only",
    "no_agreement",
    "nsa_primary",
    "direction_flip",
    "one_week_delay",
    "four_week_placebo",
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
        raise ValueError("H8DM-1 preregistration file changed")
    registration = json.loads(PREREGISTRATION.read_text())
    prereg.validate_manifest(registration)
    if registration.get("outcomes_opened") is not False:
        raise ValueError("H8DM-1 preregistration opened outcomes")
    if registration.get("opened_outcome_windows") != []:
        raise ValueError("H8DM-1 preregistration opened an outcome window")
    return cast(dict[str, Any], registration)


def _event_row(
    control: str, event: clock.Event, *, side: int | None = None
) -> dict[str, Any]:
    row = asdict(event)
    row["control"] = control
    if side is not None:
        row["side"] = side
    return {column: row[column] for column in CLOCK_COLUMNS}


def build_clock_rows(source: Any) -> list[dict[str, Any]]:
    quantile = prereg.Policy.tail_quantile
    primary = clock.build_events(source, tail_quantile=quantile)
    controls: dict[str, list[clock.Event]] = {
        "primary": primary,
        "migration_only": clock.build_events(
            source, mode="migration_only", tail_quantile=quantile
        ),
        "borrowings_only": clock.build_events(
            source, mode="borrowings_only", tail_quantile=quantile
        ),
        "cash_only": clock.build_events(
            source, mode="cash_only", tail_quantile=quantile
        ),
        "no_agreement": clock.build_events(
            source, mode="no_agreement", tail_quantile=quantile
        ),
        "nsa_primary": clock.build_events(
            source, mode="nsa_primary", tail_quantile=quantile
        ),
        "one_week_delay": clock.build_events(
            source, tail_quantile=quantile, delay_weeks=1
        ),
        "four_week_placebo": clock.build_events(
            source, tail_quantile=quantile, delay_weeks=4
        ),
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
    delay = 1 if name == "one_week_delay" else 4 if name == "four_week_placebo" else 0
    ordered = sorted(rows, key=lambda row: row["entry_time"])
    if any(row["side"] not in (-1, 1) for row in ordered):
        return False
    for row in ordered:
        release = clock.pd.Timestamp(row["signal_time"]).to_pydatetime()
        expected_entry, expected_exit = clock._execution_times(
            release, delay_weeks=delay
        )
        if row["entry_time"] != expected_entry.isoformat():
            return False
        if row["exit_time"] != expected_exit.isoformat():
            return False
        if row["release_date"] in clock.STRUCTURAL_EXCLUSION_RELEASES:
            return False
    return all(
        current["entry_time"] >= previous["exit_time"]
        for previous, current in zip(ordered, ordered[1:])
    )


def _verify_primary_clock(primary: list[clock.Event]) -> None:
    if _sha256(PRIMARY_CLOCK) != PRIMARY_CLOCK_SHA256:
        raise ValueError("H8DM-1 preregistered primary clock changed")
    expected = clock.events_frame(primary).reset_index(drop=True)
    actual = pd.read_csv(PRIMARY_CLOCK, compression="gzip")
    for column in ("signal_time", "entry_time", "exit_time"):
        actual[column] = pd.to_datetime(actual[column], utc=True, errors="raise")
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError("H8DM-1 primary clock no longer replays") from exc


def build_report(
    *, clocks_path: str | Path = DEFAULT_CLOCKS, write_clock: bool = True
) -> dict[str, Any]:
    registration = _load_registration()
    source = clock.load_source()
    primary_events = clock.build_events(source, tail_quantile=prereg.Policy.tail_quantile)
    _verify_primary_clock(primary_events)
    ledger = build_clock_rows(source)
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
        "source_rows_exactly_365": len(source) == 365,
        "selected_tail_q50_replays": prereg.select_tail_quantile(source)[0] == 0.50,
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
        "stage1_month_concentration": primary["stage1"]["max_single_month_count"]
        <= support["stage1_max_single_month_count"],
        "sealed_2023_events": primary["2023"]["events"]
        >= support["sealed_2023_events_min"],
        "sealed_2023_direction_support": (
            primary["2023"]["longs"] >= support["sealed_2023_long_min"]
            and primary["2023"]["shorts"] >= support["sealed_2023_short_min"]
        ),
        "each_sealed_2023_half": min(
            primary[half]["events"] for half in ("2023_h1", "2023_h2")
        )
        >= support["each_sealed_2023_half_min"],
        "sealed_2023_month_concentration": primary["2023"]["max_single_month_count"]
        <= support["sealed_2023_max_single_month_count"],
        "structural_break_releases_excluded": not (
            set(clock.STRUCTURAL_EXCLUSION_RELEASES)
            & {row["release_date"] for row in ledger}
        ),
        "direction_flip_is_exact": all(
            flipped["entry_time"] == original["entry_time"]
            and flipped["exit_time"] == original["exit_time"]
            and flipped["side"] == -original["side"]
            for original, flipped in zip(
                grouped["primary"], grouped["direction_flip"]
            )
        ),
        "market_or_funding_rows_opened_zero": True,
    }
    passed = all(checks.values())
    core: dict[str, Any] = {
        "protocol_version": "fed_h8_deposit_migration_support_v1",
        "policy_id": "H8DM-1",
        "as_of_date": prereg.AS_OF_DATE,
        "preregistration_commit": "e77d57f",
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
            "source_rows": len(source),
        },
        "clock_summaries": summaries,
        "entry_clock_jaccard_vs_primary": {
            control: {
                "stage1": _jaccard(
                    grouped["primary"], rows, *prereg.WINDOWS["stage1"]
                ),
                "stage2": _jaccard(
                    grouped["primary"], rows, *prereg.WINDOWS["2023"]
                ),
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


def validate_report(payload: Mapping[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("H8DM-1 support manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("H8DM-1 support opened outcomes")
    if payload.get("outcome_sources_opened") != []:
        raise ValueError("H8DM-1 support opened an outcome source")
    if payload.get("market_or_funding_rows_opened") != 0:
        raise ValueError("H8DM-1 support opened execution rows")
    if payload.get("support_passed") is not True:
        raise ValueError("H8DM-1 source support failed")


def render_docs(report: Mapping[str, Any]) -> str:
    primary = report["clock_summaries"]["primary"]
    rows = "\n".join(
        f"| {name} | {item['events']} | {item['longs']} | {item['shorts']} |"
        for name, item in primary.items()
    )
    checks = "\n".join(
        f"| `{name}` | {'PASS' if value else 'FAIL'} |"
        for name, value in report["support_checks"].items()
    )
    controls = "\n".join(
        f"| `{name}` | {count} |"
        for name, count in report["clocks"]["counts"].items()
    )
    return f"""# H8DM-1 source-support freeze — 2026-07-18

## Decision

**{report['disposition']}**

No BTC OHLC, funding, return, or existing-alpha overlap source was opened.
The primary and control clocks are now immutable.

## Primary distribution

| Window | Events | Long | Short |
|---|---:|---:|---:|
{rows}

## Checks

| Check | Result |
|---|:---:|
{checks}

## Frozen control clocks

| Control | Events |
|---|---:|
{controls}

The mechanism controls are each component alone, the composite without the
two-of-three agreement rule, and the exact not-seasonally-adjusted replay. The
falsification controls are an exact direction flip, one-week delay, and
four-week placebo.

## Identity

- preregistration SHA-256: `{report['preregistration_sha256']}`
- clock ledger SHA-256: `{report['clocks']['sha256']}`
- support manifest: `{report['manifest_hash']}`
- market/funding rows opened: `{report['market_or_funding_rows_opened']}`
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
    print(
        json.dumps(
            run(output=args.output, clocks_path=args.clocks, docs=args.docs),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
