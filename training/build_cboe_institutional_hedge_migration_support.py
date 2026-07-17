"""Freeze outcome-blind CIHM-1 support and source-control clocks."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from training import cboe_institutional_hedge_migration_clock as clock
from training import preregister_cboe_institutional_hedge_migration as prereg


PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = "0709c7aff57dc1e1e7079979ec44ceb0e154c47898ea593f2bfe50d1ab4052d5"
PRIMARY_CLOCK = Path(clock.DEFAULT_OUTPUT)
PRIMARY_CLOCK_SHA256 = "188196f1ea8d6ecd741306419e540b9ec9c11800d9b96d3d2ad591cc3fc94cf0"
DEFAULT_OUTPUT = "results/cboe_institutional_hedge_migration_support_2026-07-18.json"
DEFAULT_LEDGER = "results/cboe_institutional_hedge_migration_clocks_2026-07-18.csv.gz"
BASE_CLOCK_NAMES = (
    "primary",
    "institutional_gap_only",
    "vix_call_pressure_only",
    "index_share_only",
    "level_composite",
    "one_release_delay",
    "seven_release_placebo",
)
LEDGER_COLUMNS = ("control", *clock.EVENT_COLUMNS)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("CIHM-1 preregistration file changed")
    payload = json.loads(PREREGISTRATION.read_text())
    prereg.validate_manifest(payload)
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CIHM-1 support cannot follow an outcome-open registration")
    return payload


def build_control_events() -> dict[str, list[clock.Event]]:
    rows = clock.read_source(prereg.SOURCE_PATH)
    return {
        "primary": clock.build_events(rows),
        "institutional_gap_only": clock.build_events(
            rows, mode="institutional_gap_only"
        ),
        "vix_call_pressure_only": clock.build_events(
            rows, mode="vix_call_pressure_only"
        ),
        "index_share_only": clock.build_events(rows, mode="index_share_only"),
        "level_composite": clock.build_events(rows, mode="level_composite"),
        "one_release_delay": clock.build_events(rows, release_delay=1),
        "seven_release_placebo": clock.build_events(rows, release_delay=7),
    }


def ledger_bytes(clocks: dict[str, list[clock.Event]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(LEDGER_COLUMNS)
    for name in BASE_CLOCK_NAMES:
        for event in clocks[name]:
            writer.writerow(
                (name, *(getattr(event, column) for column in clock.EVENT_COLUMNS))
            )
    return output.getvalue().encode()


def write_gzip(path: str | Path, payload: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as handle:
            handle.write(payload)


def _window(events: list[clock.Event], start: str, end: str) -> list[clock.Event]:
    return [event for event in events if start <= event.entry_time < end]


def _distribution(events: list[clock.Event], start: str, end: str) -> dict[str, Any]:
    selected = _window(events, start, end)
    months: dict[str, int] = {}
    for event in selected:
        month = event.entry_time[:7]
        months[month] = months.get(month, 0) + 1
    maximum = max(months.values(), default=0)
    return {
        "events": len(selected),
        "longs": sum(event.side == "LONG" for event in selected),
        "shorts": sum(event.side == "SHORT" for event in selected),
        "months": len(months),
        "max_single_month_count": maximum,
        "max_single_month_share": maximum / len(selected) if selected else 0.0,
    }


def _jaccard(left: list[clock.Event], right: list[clock.Event]) -> float:
    a = {event.entry_time for event in left}
    b = {event.entry_time for event in right}
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _verify_primary_clock(primary: list[clock.Event]) -> None:
    if sha256_file(PRIMARY_CLOCK) != PRIMARY_CLOCK_SHA256:
        raise RuntimeError("CIHM-1 preregistered primary clock changed")
    expected = clock.event_csv(primary)
    with gzip.open(PRIMARY_CLOCK, "rb") as handle:
        actual = handle.read()
    if actual != expected:
        raise RuntimeError("CIHM-1 primary clock does not replay from frozen source")


def build_support(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    ledger_path: str | Path = DEFAULT_LEDGER,
) -> dict[str, Any]:
    registration = _load_preregistration()
    clocks = build_control_events()
    _verify_primary_clock(clocks["primary"])
    ledger = ledger_bytes(clocks)
    write_gzip(ledger_path, ledger)
    distributions = {
        name: {
            window: _distribution(events, *bounds)
            for window, bounds in prereg.WINDOWS.items()
        }
        for name, events in clocks.items()
    }
    primary = distributions["primary"]
    policy = registration["support_freeze_before_returns"]
    checks = {
        "stage1_events": primary["stage1"]["events"] >= policy["stage1_events_min"],
        "each_stage1_year": (
            min(primary[year]["events"] for year in ("2021", "2022"))
            >= policy["each_stage1_year_min"]
        ),
        "sealed_2023_events": (
            primary["2023"]["events"] >= policy["sealed_2023_events_min"]
        ),
        "each_sealed_2023_half": (
            min(primary[half]["events"] for half in ("2023_h1", "2023_h2"))
            >= policy["each_sealed_2023_half_min"]
        ),
        "month_concentration": (
            max(
                primary["stage1"]["max_single_month_share"],
                primary["2023"]["max_single_month_share"],
            )
            <= policy["maximum_single_month_share"]
        ),
        "short_only": all(event.side == "SHORT" for event in clocks["primary"]),
    }
    support_passed = all(checks.values())
    core: dict[str, Any] = {
        "protocol_version": "cboe_institutional_hedge_migration_support_v1",
        "as_of_date": "2026-07-18",
        "policy_id": "CIHM-1",
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "source": {
            "panel": prereg.SOURCE_PATH,
            "panel_sha256": sha256_file(prereg.SOURCE_PATH),
            "manifest": prereg.SOURCE_MANIFEST,
            "manifest_sha256": sha256_file(prereg.SOURCE_MANIFEST),
            "columns_loaded": list(clock.SOURCE_COLUMNS),
            "source_rows": len(clock.read_source(prereg.SOURCE_PATH)),
        },
        "clocks": {
            "path": str(ledger_path),
            "sha256": sha256_file(ledger_path),
            "rows": sum(len(events) for events in clocks.values()),
            "counts": {name: len(events) for name, events in clocks.items()},
            "distributions": distributions,
            "entry_clock_jaccard_vs_primary": {
                name: _jaccard(clocks["primary"], events)
                for name, events in clocks.items()
                if name != "primary"
            },
        },
        "checks": checks,
        "support_passed": support_passed,
        "advance_to_stage1_outcomes": support_passed,
        "forbidden_opened": {
            "BTC_OHLC": False,
            "funding": False,
            "returns": False,
            "existing_alpha_overlap": False,
        },
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def validate_support(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("CIHM-1 support manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CIHM-1 support opened outcomes")
    if payload.get("outcome_sources_opened") != []:
        raise RuntimeError("CIHM-1 support opened an outcome source")
    if payload.get("support_passed") is not True:
        raise RuntimeError("CIHM-1 source support failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    args = parser.parse_args()
    payload = build_support(output_path=args.output, ledger_path=args.ledger)
    validate_support(payload)
    print(
        json.dumps(
            {
                "outcomes_opened": False,
                "policy_id": payload["policy_id"],
                "support_passed": payload["support_passed"],
                "manifest_hash": payload["manifest_hash"],
                "ledger_sha256": payload["clocks"]["sha256"],
                "output": args.output,
                "ledger": args.ledger,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
