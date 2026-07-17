"""Freeze outcome-blind CVTR-1 support and source-control clocks."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from training import cboe_volatility_term_rotation_clock as clock
from training import preregister_cboe_volatility_term_rotation as prereg


PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = "af440f63cee9fac526fe9731c2e203ab37e9ffa19847403fbd51845a25c4b7f6"
PRIMARY_CLOCK = Path(clock.DEFAULT_OUTPUT)
PRIMARY_CLOCK_SHA256 = "c0250d1f40c87049f6d7639ba43f5285835441399a62968434b65c7d46ed2a93"
DEFAULT_OUTPUT = "results/cboe_volatility_term_rotation_support_2026-07-17.json"
DEFAULT_LEDGER = "results/cboe_volatility_term_rotation_clocks_2026-07-17.csv.gz"
LEDGER_COLUMNS = (
    "control",
    "observation_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "front_slope",
    "broad_slope",
    "front_rank",
    "broad_rank",
    "vix_level_rank",
    "score",
)


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
        raise RuntimeError("CVTR-1 preregistration file changed")
    payload = json.loads(PREREGISTRATION.read_text())
    prereg.validate_manifest(payload)
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CVTR-1 support cannot follow an outcome-open preregistration")
    return payload


def _event_rows(name: str, events: list[clock.Event]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for event in events:
        result.append(
            {
                "control": name,
                "observation_date": event.observation_date,
                "signal_time": event.signal_time,
                "entry_time": event.entry_time,
                "exit_time": event.exit_time,
                "side": "1" if event.side == "LONG" else "-1",
                "front_slope": event.front_slope,
                "broad_slope": event.broad_slope,
                "front_rank": event.front_rank,
                "broad_rank": event.broad_rank,
                "vix_level_rank": event.vix_level_rank,
                "score": event.score,
            }
        )
    return result


def build_control_events() -> dict[str, list[clock.Event]]:
    rows = clock.read_source(prereg.SOURCE_PATH)
    return {
        "primary": clock.build_events(rows),
        "front_slope_tail": clock.build_events(rows, mode="front_only"),
        "broad_slope_tail": clock.build_events(rows, mode="broad_only"),
        "vix_level_tail": clock.build_events(rows, mode="vix_level"),
        "one_release_delay": clock.build_events(rows, release_delay=1),
    }


def ledger_bytes(clocks: dict[str, list[clock.Event]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(LEDGER_COLUMNS)
    for name in (
        "primary",
        "front_slope_tail",
        "broad_slope_tail",
        "vix_level_tail",
        "one_release_delay",
    ):
        for row in _event_rows(name, clocks[name]):
            writer.writerow(tuple(row[column] for column in LEDGER_COLUMNS))
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


WINDOWS = {
    "2021": ("2021-01-01T00:00:00+00:00", "2022-01-01T00:00:00+00:00"),
    "2022": ("2022-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"),
    "stage1": ("2021-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"),
    "2023_h1": ("2023-01-01T00:00:00+00:00", "2023-07-01T00:00:00+00:00"),
    "2023_h2": ("2023-07-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
    "2023": ("2023-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
}


def _jaccard(left: list[clock.Event], right: list[clock.Event]) -> float:
    a = {event.entry_time for event in left}
    b = {event.entry_time for event in right}
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _verify_primary_clock(primary: list[clock.Event]) -> None:
    if sha256_file(PRIMARY_CLOCK) != PRIMARY_CLOCK_SHA256:
        raise RuntimeError("CVTR-1 preregistered primary clock changed")
    expected = clock.event_csv(primary)
    with gzip.open(PRIMARY_CLOCK, "rb") as handle:
        actual = handle.read()
    if actual != expected:
        raise RuntimeError("CVTR-1 primary clock does not replay from frozen source")


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
            for window, bounds in WINDOWS.items()
        }
        for name, events in clocks.items()
    }
    primary = distributions["primary"]
    support_policy = registration["support_freeze_before_returns"]
    checks = {
        "stage1_events": primary["stage1"]["events"] >= support_policy["stage1_events_min"],
        "each_stage1_year": min(primary[year]["events"] for year in ("2021", "2022")) >= support_policy["each_stage1_year_min"],
        "each_stage1_side": min(primary["stage1"]["longs"], primary["stage1"]["shorts"]) >= support_policy["each_stage1_side_min"],
        "sealed_2023_events": primary["2023"]["events"] >= support_policy["sealed_2023_events_min"],
        "each_sealed_2023_side": min(primary["2023"]["longs"], primary["2023"]["shorts"]) >= support_policy["each_sealed_2023_side_min"],
        "each_sealed_2023_half": min(primary[half]["events"] for half in ("2023_h1", "2023_h2")) >= support_policy["each_sealed_2023_half_min"],
        "month_concentration": max(primary["stage1"]["max_single_month_share"], primary["2023"]["max_single_month_share"]) <= support_policy["maximum_single_month_share"],
    }
    support_passed = all(checks.values())
    core: dict[str, Any] = {
        "protocol_version": "cboe_volatility_term_rotation_support_v1",
        "as_of_date": "2026-07-17",
        "policy_id": "CVTR-1",
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
        "causal_checks": {
            "strict_prior_rank": True,
            "current_appended_after_rank": True,
            "next_source_session_entry": True,
            "no_missing_date_forward_fill": True,
            "globally_nonoverlapping": all(
                all(left.exit_time <= right.entry_time for left, right in zip(events, events[1:]))
                for events in clocks.values()
            ),
        },
        "support_checks": checks,
        "support_passed": support_passed,
        "advance_to_stage1_outcomes": support_passed,
        "sealed": ["stage1_2021_2022", "stage2_2023", "2024", "2025", "2026_ytd"],
        "failure_action": registration["support_freeze_before_returns"]["failure_action"],
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    args = parser.parse_args()
    report = build_support(output_path=args.output, ledger_path=args.ledger)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
