"""Build the exact source-only SFRD-1 event clock.

SOFR rates are parsed as base-10 decimals and converted to integer basis
points before differencing.  The empirical mid-rank is represented by an
integer numerator over ``2 * lookback``.  No binary floating-point comparison
and no crypto market field enters this module.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SOURCE_PATH = (
    "data/new_york_fed_sofr_distribution_2018_2023/"
    "new_york_fed_sofr_distribution_2018-04-02_2023-12-28.csv.gz"
)
DEFAULT_OUTPUT = "results/sofr_rate_dislocation_preregistered_clock_2026-07-17.csv.gz"
REQUIRED_COLUMNS = (
    "effective_date",
    "sofr_available_at_utc",
    "sofr_percent",
)
OUTPUT_COLUMNS = (
    "event_index",
    "effective_date",
    "sofr_available_at_utc",
    "delta_bp",
    "rank_twice_numerator",
    "rank_twice_denominator",
    "state",
    "side",
    "entry_time",
    "exit_time",
)


@dataclass(frozen=True)
class ClockConfig:
    source: str = SOURCE_PATH
    output: str = DEFAULT_OUTPUT
    lookback_observations: int = 120
    lower_rank_twice_numerator_max: int = 36
    upper_rank_twice_numerator_min: int = 204
    bar_minutes: int = 5
    execution_delay_bars: int = 1
    hold_bars: int = 1440


@dataclass(frozen=True)
class SourceRow:
    effective_date: date
    available_at: datetime
    rate_bp: int


@dataclass(frozen=True)
class Event:
    event_index: int
    effective_date: str
    sofr_available_at_utc: str
    delta_bp: int
    rank_twice_numerator: int
    rank_twice_denominator: int
    state: int
    side: int
    entry_time: str
    exit_time: str


def _parse_rate_bp(value: str) -> int:
    try:
        scaled = Decimal(value) * Decimal(100)
    except InvalidOperation as exc:
        raise ValueError(f"SOFR rate is not a base-10 decimal: {value!r}") from exc
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise ValueError(f"SOFR rate is not an exact integer basis point: {value!r}")
    return int(integral)


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp: {value!r}") from exc
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"SOFR availability must be UTC: {value!r}")
    return parsed.astimezone(timezone.utc)


def read_source(path: str | Path = SOURCE_PATH) -> list[SourceRow]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(REQUIRED_COLUMNS).issubset(
            reader.fieldnames
        ):
            raise ValueError("SOFR source is missing required clock columns")
        rows = [
            SourceRow(
                effective_date=date.fromisoformat(raw["effective_date"]),
                available_at=_parse_utc(raw["sofr_available_at_utc"]),
                rate_bp=_parse_rate_bp(raw["sofr_percent"]),
            )
            for raw in reader
        ]
    if len(rows) < 2:
        raise ValueError("SOFR source needs at least two rows")
    dates = [row.effective_date for row in rows]
    availability = [row.available_at for row in rows]
    if dates != sorted(set(dates)):
        raise ValueError("SOFR effective dates must be unique and increasing")
    if availability != sorted(set(availability)):
        raise ValueError("SOFR availability timestamps must be unique and increasing")
    return rows


def build_events(
    rows: list[SourceRow], cfg: ClockConfig | None = None
) -> list[Event]:
    cfg = cfg or ClockConfig()
    denominator = 2 * cfg.lookback_observations
    if not (
        0 <= cfg.lower_rank_twice_numerator_max < denominator / 2
        and denominator / 2 < cfg.upper_rank_twice_numerator_min <= denominator
    ):
        raise ValueError("rank numerator thresholds must define lower and upper tails")
    if cfg.execution_delay_bars < 1 or cfg.hold_bars < 1 or cfg.bar_minutes < 1:
        raise ValueError("clock delays and hold must be positive")

    deltas: list[int | None] = [None]
    deltas.extend(
        row.rate_bp - prior.rate_bp for prior, row in zip(rows, rows[1:])
    )
    states = [0] * len(rows)
    rank_numerators: list[int | None] = [None] * len(rows)
    for index, current in enumerate(deltas):
        if current is None or index < cfg.lookback_observations + 1:
            continue
        prior = deltas[index - cfg.lookback_observations : index]
        if len(prior) != cfg.lookback_observations or any(
            value is None for value in prior
        ):
            raise ValueError("SFRD rank window is not exactly complete")
        integer_prior = [int(value) for value in prior if value is not None]
        numerator = 2 * sum(value < current for value in integer_prior) + sum(
            value == current for value in integer_prior
        )
        rank_numerators[index] = numerator
        if numerator >= cfg.upper_rank_twice_numerator_min:
            states[index] = 1
        elif numerator <= cfg.lower_rank_twice_numerator_max:
            states[index] = -1

    delay = timedelta(minutes=cfg.bar_minutes * cfg.execution_delay_bars)
    hold = timedelta(minutes=cfg.bar_minutes * cfg.hold_bars)
    reserved_until = datetime.min.replace(tzinfo=timezone.utc)
    events: list[Event] = []
    for index, state in enumerate(states):
        previous_state = states[index - 1] if index else 0
        if state == 0 or state == previous_state:
            continue
        entry = rows[index].available_at + delay
        if entry < reserved_until:
            continue
        exit_time = entry + hold
        numerator = rank_numerators[index]
        current_delta = deltas[index]
        if numerator is None or current_delta is None:
            raise AssertionError("admitted event lacks exact source features")
        events.append(
            Event(
                event_index=len(events),
                effective_date=rows[index].effective_date.isoformat(),
                sofr_available_at_utc=rows[index].available_at.isoformat(),
                delta_bp=current_delta,
                rank_twice_numerator=numerator,
                rank_twice_denominator=denominator,
                state=state,
                side=-state,
                entry_time=entry.isoformat(),
                exit_time=exit_time.isoformat(),
            )
        )
        reserved_until = exit_time
    return events


def contained_events(events: list[Event], start: str, end: str) -> list[Event]:
    lower = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    upper = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    return [
        event
        for event in events
        if _parse_utc(event.entry_time) >= lower and _parse_utc(event.exit_time) <= upper
    ]


def event_summary(events: list[Event], start: str, end: str) -> dict[str, Any]:
    selected = contained_events(events, start, end)
    month_counts: dict[str, int] = {}
    for event in selected:
        month = event.entry_time[:7]
        month_counts[month] = month_counts.get(month, 0) + 1
    count = len(selected)
    return {
        "count": count,
        "long": sum(event.side == 1 for event in selected),
        "short": sum(event.side == -1 for event in selected),
        "max_single_month_count": max(month_counts.values(), default=0),
        "max_single_month_share": (
            max(month_counts.values(), default=0) / count if count else None
        ),
    }


def read_event_ledger(path: str | Path = DEFAULT_OUTPUT) -> list[Event]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
            raise ValueError("SFRD event ledger columns changed")
        events = [
            Event(
                event_index=int(raw["event_index"]),
                effective_date=raw["effective_date"],
                sofr_available_at_utc=raw["sofr_available_at_utc"],
                delta_bp=int(raw["delta_bp"]),
                rank_twice_numerator=int(raw["rank_twice_numerator"]),
                rank_twice_denominator=int(raw["rank_twice_denominator"]),
                state=int(raw["state"]),
                side=int(raw["side"]),
                entry_time=raw["entry_time"],
                exit_time=raw["exit_time"],
            )
            for raw in reader
        ]
    if [event.event_index for event in events] != list(range(len(events))):
        raise ValueError("SFRD event ledger indices are not exact and contiguous")
    for event in events:
        date.fromisoformat(event.effective_date)
        _parse_utc(event.sofr_available_at_utc)
        _parse_utc(event.entry_time)
        _parse_utc(event.exit_time)
        if event.state not in (-1, 1) or event.side != -event.state:
            raise ValueError("SFRD event ledger side/state mapping changed")
    return events


def _write_events(path: Path, events: list[Event]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=list(OUTPUT_COLUMNS), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(asdict(event) for event in events)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_clock(cfg: ClockConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ClockConfig()
    events = build_events(read_source(cfg.source), cfg)
    output = Path(cfg.output)
    digest = _write_events(output, events)
    return {
        "config": asdict(cfg),
        "output": str(output),
        "sha256": digest,
        "events": len(events),
        "outcomes_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cfg = ClockConfig(source=args.source, output=args.output)
    print(json.dumps(write_clock(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
