"""Build the outcome-blind Treasury Auction Demand Impulse source clock."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Literal, cast


DEFAULT_SOURCE = (
    "data/us_treasury_auction_demand_2016_2023/"
    "us_treasury_nominal_original_auctions_2016_2023.csv.gz"
)
DEFAULT_OUTPUT = (
    "results/treasury_auction_demand_impulse_preregistered_clock_2026-07-17.csv.gz"
)
SOURCE_COLUMNS = (
    "auction_date",
    "result_available_at_utc",
    "original_security_term",
    "cusip",
    "bid_to_cover_ratio",
    "indirect_competitive_share",
    "source_complete",
)
TERM_PRIORITY = {
    "2-Year": 2,
    "3-Year": 3,
    "5-Year": 5,
    "7-Year": 7,
    "10-Year": 10,
    "20-Year": 20,
    "30-Year": 30,
}
EVENT_COLUMNS = (
    "auction_date",
    "decision_time",
    "entry_time",
    "scheduled_exit_time",
    "original_security_term",
    "cusip",
    "side",
    "clock_mode",
    "bid_to_cover_delta",
    "indirect_share_delta",
    "bid_to_cover_delta_rank",
    "indirect_share_delta_rank",
)
ClockMode = Literal["primary", "bid_to_cover_only", "indirect_only"]


@dataclass(frozen=True)
class SourceRow:
    auction_date: str
    available_at: datetime
    term: str
    cusip: str
    bid_to_cover: Decimal | None
    indirect_share: Decimal | None
    source_complete: bool


@dataclass(frozen=True)
class ChangeRow:
    auction_date: str
    available_at: datetime
    term: str
    cusip: str
    bid_to_cover_delta: Decimal
    indirect_share_delta: Decimal
    bid_to_cover_delta_rank: float | None
    indirect_share_delta_rank: float | None


@dataclass(frozen=True)
class Event:
    auction_date: str
    decision_time: str
    entry_time: str
    scheduled_exit_time: str
    original_security_term: str
    cusip: str
    side: str
    clock_mode: str
    bid_to_cover_delta: str
    indirect_share_delta: str
    bid_to_cover_delta_rank: str
    indirect_share_delta_rank: str


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid source UTC timestamp: {value!r}") from exc
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"source timestamp must be UTC: {value!r}")
    return parsed


def _parse_decimal(value: str, *, field: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{field} is not base-10 numeric: {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field} must be finite")
    return parsed


def read_source(path: str | Path = DEFAULT_SOURCE) -> list[SourceRow]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(SOURCE_COLUMNS).issubset(
            reader.fieldnames
        ):
            raise ValueError("Treasury auction source is missing required columns")
        rows: list[SourceRow] = []
        for raw in reader:
            term = raw["original_security_term"]
            if term not in TERM_PRIORITY:
                raise ValueError(f"unexpected original term: {term!r}")
            complete = raw["source_complete"] == "true"
            bid_to_cover = None
            indirect_share = None
            if complete:
                bid_to_cover = _parse_decimal(
                    raw["bid_to_cover_ratio"], field="bid_to_cover_ratio"
                )
                indirect_share = _parse_decimal(
                    raw["indirect_competitive_share"],
                    field="indirect_competitive_share",
                )
            elif raw["bid_to_cover_ratio"] or raw["indirect_competitive_share"]:
                raise ValueError("incomplete source row exposes quarantined demand values")
            rows.append(
                SourceRow(
                    auction_date=raw["auction_date"],
                    available_at=_parse_utc(raw["result_available_at_utc"]),
                    term=term,
                    cusip=raw["cusip"],
                    bid_to_cover=bid_to_cover,
                    indirect_share=indirect_share,
                    source_complete=complete,
                )
            )
    if not rows:
        raise ValueError("Treasury auction source is empty")
    if any(left.available_at > right.available_at for left, right in zip(rows, rows[1:])):
        raise ValueError("Treasury auction source is not chronologically ordered")
    ordered = sorted(
        rows, key=lambda row: (row.available_at, TERM_PRIORITY[row.term], row.cusip)
    )
    keys = [(row.auction_date, row.cusip) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Treasury auction source contains duplicate rows")
    return ordered


def strict_prior_midrank(current: Decimal, prior: list[Decimal]) -> float:
    if not prior:
        raise ValueError("strict-prior rank needs history")
    less = sum(value < current for value in prior)
    equal = sum(value == current for value in prior)
    return (less + 0.5 * equal) / len(prior)


def build_changes(
    rows: Iterable[SourceRow], *, prior_changes: int = 12
) -> list[ChangeRow]:
    if prior_changes < 2:
        raise ValueError("prior_changes must be at least two")
    previous: dict[str, SourceRow] = {}
    histories: dict[str, list[tuple[Decimal, Decimal]]] = defaultdict(list)
    changes: list[ChangeRow] = []
    for row in rows:
        if not row.source_complete:
            # Do not bridge a current demand change across a quarantined auction.
            previous.pop(row.term, None)
            continue
        prior_row = previous.get(row.term)
        if prior_row is not None:
            assert row.bid_to_cover is not None
            assert row.indirect_share is not None
            assert prior_row.bid_to_cover is not None
            assert prior_row.indirect_share is not None
            bid_delta = row.bid_to_cover - prior_row.bid_to_cover
            indirect_delta = row.indirect_share - prior_row.indirect_share
            history = histories[row.term]
            bid_rank = None
            indirect_rank = None
            if len(history) >= prior_changes:
                window = history[-prior_changes:]
                bid_rank = strict_prior_midrank(
                    bid_delta, [item[0] for item in window]
                )
                indirect_rank = strict_prior_midrank(
                    indirect_delta, [item[1] for item in window]
                )
            changes.append(
                ChangeRow(
                    auction_date=row.auction_date,
                    available_at=row.available_at,
                    term=row.term,
                    cusip=row.cusip,
                    bid_to_cover_delta=bid_delta,
                    indirect_share_delta=indirect_delta,
                    bid_to_cover_delta_rank=bid_rank,
                    indirect_share_delta_rank=indirect_rank,
                )
            )
            history.append((bid_delta, indirect_delta))
        previous[row.term] = row
    return changes


def _side(change: ChangeRow, *, mode: ClockMode, threshold: float) -> str | None:
    bid_rank = change.bid_to_cover_delta_rank
    indirect_rank = change.indirect_share_delta_rank
    if bid_rank is None or indirect_rank is None:
        return None
    lower = 1.0 - threshold
    if mode == "primary":
        if bid_rank >= threshold and indirect_rank >= threshold:
            return "LONG"
        if bid_rank <= lower and indirect_rank <= lower:
            return "SHORT"
    elif mode == "bid_to_cover_only":
        if bid_rank >= threshold:
            return "LONG"
        if bid_rank <= lower:
            return "SHORT"
    elif mode == "indirect_only":
        if indirect_rank >= threshold:
            return "LONG"
        if indirect_rank <= lower:
            return "SHORT"
    else:
        raise ValueError(f"unsupported clock mode: {mode}")
    return None


def build_events(
    rows: Iterable[SourceRow],
    *,
    mode: ClockMode = "primary",
    prior_changes: int = 12,
    threshold: float = 0.75,
    execution_delay_minutes: int = 5,
    hold_hours: int = 24,
) -> list[Event]:
    if not 0.5 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between 0.5 and 1")
    if execution_delay_minutes < 1 or hold_hours < 1:
        raise ValueError("execution delay and hold must be positive")
    candidates: list[tuple[datetime, datetime, ChangeRow, str]] = []
    for change in build_changes(rows, prior_changes=prior_changes):
        side = _side(change, mode=mode, threshold=threshold)
        if side is None:
            continue
        entry = change.available_at + timedelta(minutes=execution_delay_minutes)
        exit_time = entry + timedelta(hours=hold_hours)
        candidates.append((entry, exit_time, change, side))

    events: list[Event] = []
    reserved_until: datetime | None = None
    for entry, exit_time, change, side in sorted(
        candidates,
        key=lambda item: (item[0], TERM_PRIORITY[item[2].term], item[2].cusip),
    ):
        if reserved_until is not None and entry < reserved_until:
            continue
        reserved_until = exit_time
        events.append(
            Event(
                auction_date=change.auction_date,
                decision_time=change.available_at.isoformat(),
                entry_time=entry.isoformat(),
                scheduled_exit_time=exit_time.isoformat(),
                original_security_term=change.term,
                cusip=change.cusip,
                side=side,
                clock_mode=mode,
                bid_to_cover_delta=format(change.bid_to_cover_delta, "f"),
                indirect_share_delta=format(change.indirect_share_delta, "f"),
                bid_to_cover_delta_rank=format(
                    change.bid_to_cover_delta_rank or 0.0, ".12g"
                ),
                indirect_share_delta_rank=format(
                    change.indirect_share_delta_rank or 0.0, ".12g"
                ),
            )
        )
    return events


def write_events(path: str | Path, events: list[Event]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=list(EVENT_COLUMNS), lineterminator="\n"
                )
                writer.writeheader()
                for event in events:
                    writer.writerow(cast(Any, asdict(event)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("primary", "bid_to_cover_only", "indirect_only"),
        default="primary",
    )
    args = parser.parse_args()
    events = build_events(read_source(args.source), mode=args.mode)
    write_events(args.output, events)
    print(f"wrote {len(events)} source-only events to {args.output}")


if __name__ == "__main__":
    main()
