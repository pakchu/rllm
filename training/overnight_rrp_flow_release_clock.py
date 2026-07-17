"""Build the outcome-blind Overnight RRP Flow Release (ORFR-1) clock."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Literal, cast


DEFAULT_SOURCE = (
    "data/new_york_fed_overnight_rrp_2018_2023/"
    "new_york_fed_overnight_rrp_2018-01-01_2023-12-31.csv.gz"
)
DEFAULT_OUTPUT = "results/overnight_rrp_flow_release_preregistered_clock_2026-07-17.csv.gz"
SOURCE_COLUMNS = (
    "operation_date",
    "result_available_at_utc",
    "total_amount_accepted_usd",
    "source_complete",
)
EVENT_COLUMNS = (
    "operation_date",
    "decision_time",
    "entry_time",
    "scheduled_exit_time",
    "side",
    "clock_mode",
    "log_amount",
    "innovation",
    "innovation_rank",
)
ClockMode = Literal["primary", "one_day_delta"]


@dataclass(frozen=True)
class SourceRow:
    operation_date: str
    available_at: datetime
    amount_usd: int | None
    source_complete: bool


@dataclass(frozen=True)
class FeatureRow:
    source_index: int
    operation_date: str
    available_at: datetime
    log_amount: float
    residual_innovation: float | None
    residual_rank: float | None
    one_day_delta: float | None
    one_day_delta_rank: float | None


@dataclass(frozen=True)
class Event:
    operation_date: str
    decision_time: str
    entry_time: str
    scheduled_exit_time: str
    side: str
    clock_mode: str
    log_amount: str
    innovation: str
    innovation_rank: str


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid UTC timestamp: {value!r}") from exc
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"timestamp must be UTC: {value!r}")
    return parsed


def _parse_amount(value: str) -> int:
    try:
        amount = int(value)
    except ValueError as exc:
        raise ValueError(f"ON RRP amount must be integer USD: {value!r}") from exc
    if amount < 0:
        raise ValueError("ON RRP amount must be nonnegative")
    return amount


def read_source(path: str | Path = DEFAULT_SOURCE) -> list[SourceRow]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(SOURCE_COLUMNS).issubset(
            reader.fieldnames
        ):
            raise ValueError("ON RRP source is missing required columns")
        rows: list[SourceRow] = []
        for raw in reader:
            complete = raw["source_complete"] == "true"
            amount = _parse_amount(raw["total_amount_accepted_usd"]) if complete else None
            if not complete and raw["total_amount_accepted_usd"]:
                raise ValueError("quarantined ON RRP row exposes an amount")
            rows.append(
                SourceRow(
                    operation_date=raw["operation_date"],
                    available_at=_parse_utc(raw["result_available_at_utc"]),
                    amount_usd=amount,
                    source_complete=complete,
                )
            )
    if not rows:
        raise ValueError("ON RRP source is empty")
    if any(left.available_at >= right.available_at for left, right in zip(rows, rows[1:])):
        raise ValueError("ON RRP source clock must be strictly increasing")
    dates = [row.operation_date for row in rows]
    if len(dates) != len(set(dates)):
        raise ValueError("ON RRP source contains duplicate dates")
    return rows


def strict_prior_midrank(current: float, prior: list[float]) -> float:
    if not prior:
        raise ValueError("strict-prior rank needs history")
    less = sum(value < current for value in prior)
    equal = sum(value == current for value in prior)
    return (less + 0.5 * equal) / len(prior)


def build_features(
    rows: Iterable[SourceRow],
    *,
    baseline_operations: int = 5,
    rank_operations: int = 104,
) -> list[FeatureRow]:
    if baseline_operations < 2:
        raise ValueError("baseline_operations must be at least two")
    if rank_operations < 20:
        raise ValueError("rank_operations must be at least 20")
    source_rows = list(rows)
    baseline: list[float] = []
    residual_history: list[float] = []
    delta_history: list[float] = []
    previous_log_amount: float | None = None
    features: list[FeatureRow] = []
    for index, row in enumerate(source_rows):
        if not row.source_complete:
            # Never bridge either local baseline across a later-updated archive row.
            baseline.clear()
            previous_log_amount = None
            continue
        assert row.amount_usd is not None
        log_amount = math.log1p(row.amount_usd / 1_000_000_000.0)

        residual = None
        residual_rank = None
        if len(baseline) >= baseline_operations:
            residual = log_amount - statistics.median(baseline[-baseline_operations:])
            if len(residual_history) >= rank_operations:
                residual_rank = strict_prior_midrank(
                    residual, residual_history[-rank_operations:]
                )
            residual_history.append(residual)

        one_day_delta = None
        one_day_delta_rank = None
        if previous_log_amount is not None:
            one_day_delta = log_amount - previous_log_amount
            if len(delta_history) >= rank_operations:
                one_day_delta_rank = strict_prior_midrank(
                    one_day_delta, delta_history[-rank_operations:]
                )
            delta_history.append(one_day_delta)

        features.append(
            FeatureRow(
                source_index=index,
                operation_date=row.operation_date,
                available_at=row.available_at,
                log_amount=log_amount,
                residual_innovation=residual,
                residual_rank=residual_rank,
                one_day_delta=one_day_delta,
                one_day_delta_rank=one_day_delta_rank,
            )
        )
        baseline.append(log_amount)
        previous_log_amount = log_amount
    return features


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("ORFR feature is not finite")
    return format(value, ".17g")


def build_events(
    rows: Iterable[SourceRow],
    *,
    mode: ClockMode = "primary",
    baseline_operations: int = 5,
    rank_operations: int = 104,
    lower_tail: float = 0.125,
) -> list[Event]:
    if not 0.0 < lower_tail < 0.5:
        raise ValueError("lower_tail must be between zero and one-half")
    source_rows = list(rows)
    features = build_features(
        source_rows,
        baseline_operations=baseline_operations,
        rank_operations=rank_operations,
    )
    events: list[Event] = []
    for feature in features:
        if feature.source_index + 1 >= len(source_rows):
            continue
        if mode == "primary":
            innovation = feature.residual_innovation
            rank = feature.residual_rank
        elif mode == "one_day_delta":
            innovation = feature.one_day_delta
            rank = feature.one_day_delta_rank
        else:
            raise ValueError(f"unknown ORFR clock mode: {mode}")
        if innovation is None or rank is None:
            continue
        if rank <= lower_tail:
            side = "LONG"
        elif rank >= 1.0 - lower_tail:
            side = "SHORT"
        else:
            continue
        entry = feature.available_at + timedelta(minutes=5)
        exit_time = source_rows[feature.source_index + 1].available_at + timedelta(
            minutes=5
        )
        if exit_time <= entry:
            raise ValueError("ORFR exit must follow entry")
        events.append(
            Event(
                operation_date=feature.operation_date,
                decision_time=feature.available_at.isoformat(),
                entry_time=entry.isoformat(),
                scheduled_exit_time=exit_time.isoformat(),
                side=side,
                clock_mode=mode,
                log_amount=_format_float(feature.log_amount),
                innovation=_format_float(innovation),
                innovation_rank=_format_float(rank),
            )
        )
    if any(
        _parse_utc(left.scheduled_exit_time) > _parse_utc(right.entry_time)
        for left, right in zip(events, events[1:])
    ):
        raise ValueError("ORFR event clock overlaps")
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
                writer.writerows(cast(Any, [asdict(event) for event in events]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("primary", "one_day_delta"), default="primary")
    args = parser.parse_args()
    events = build_events(read_source(args.source), mode=cast(ClockMode, args.mode))
    write_events(args.output, events)
    print(f"wrote {len(events)} source-only ORFR events to {args.output}")


if __name__ == "__main__":
    main()
