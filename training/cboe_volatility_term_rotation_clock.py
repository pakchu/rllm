"""Build outcome-blind Cboe Volatility Term Rotation (CVTR-1) clocks."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Iterable, Literal, cast
from zoneinfo import ZoneInfo


DEFAULT_SOURCE = (
    "data/cboe_volatility_term_structure_2018_2023/"
    "cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz"
)
DEFAULT_OUTPUT = (
    "results/cboe_volatility_term_rotation_preregistered_clock_2026-07-17.csv.gz"
)
SOURCE_COLUMNS = ("observation_date", "VIX9D_close", "VIX_close", "VIX3M_close")
EVENT_COLUMNS = (
    "observation_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "clock_mode",
    "front_slope",
    "broad_slope",
    "front_rank",
    "broad_rank",
    "vix_level_rank",
    "score",
)
ClockMode = Literal["primary", "front_only", "broad_only", "vix_level"]
NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SourceRow:
    observation_date: date
    vix9d_close: float
    vix_close: float
    vix3m_close: float


@dataclass(frozen=True)
class FeatureRow:
    source_index: int
    observation_date: date
    front_slope: float
    broad_slope: float
    front_rank: float | None
    broad_rank: float | None
    vix_level_rank: float | None


@dataclass(frozen=True)
class Event:
    observation_date: str
    signal_time: str
    entry_time: str
    exit_time: str
    side: str
    clock_mode: str
    front_slope: str
    broad_slope: str
    front_rank: str
    broad_rank: str
    vix_level_rank: str
    score: str


def _positive(value: str, *, field: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def read_source(path: str | Path = DEFAULT_SOURCE) -> list[SourceRow]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError("Cboe term panel schema changed")
        rows = [
            SourceRow(
                observation_date=date.fromisoformat(raw["observation_date"]),
                vix9d_close=_positive(raw["VIX9D_close"], field="VIX9D_close"),
                vix_close=_positive(raw["VIX_close"], field="VIX_close"),
                vix3m_close=_positive(raw["VIX3M_close"], field="VIX3M_close"),
            )
            for raw in reader
        ]
    if not rows:
        raise ValueError("Cboe term panel is empty")
    dates = [row.observation_date for row in rows]
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("Cboe observation dates must be strictly increasing")
    if dates[-1] >= date(2024, 1, 1):
        raise ValueError("Cboe source escaped the frozen pre-2024 horizon")
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
    lookback_observations: int = 252,
    minimum_history: int = 126,
) -> list[FeatureRow]:
    if lookback_observations < minimum_history or minimum_history < 3:
        raise ValueError("invalid CVTR rank history")
    source_rows = list(rows)
    front_history: list[float] = []
    broad_history: list[float] = []
    vix_history: list[float] = []
    result: list[FeatureRow] = []
    for index, row in enumerate(source_rows):
        front = math.log(row.vix9d_close / row.vix_close)
        broad = math.log(row.vix_close / row.vix3m_close)
        vix_level = math.log(row.vix_close)
        front_prior = front_history[-lookback_observations:]
        broad_prior = broad_history[-lookback_observations:]
        vix_prior = vix_history[-lookback_observations:]
        front_rank = (
            strict_prior_midrank(front, front_prior)
            if len(front_prior) >= minimum_history
            else None
        )
        broad_rank = (
            strict_prior_midrank(broad, broad_prior)
            if len(broad_prior) >= minimum_history
            else None
        )
        vix_rank = (
            strict_prior_midrank(vix_level, vix_prior)
            if len(vix_prior) >= minimum_history
            else None
        )
        result.append(
            FeatureRow(
                source_index=index,
                observation_date=row.observation_date,
                front_slope=front,
                broad_slope=broad,
                front_rank=front_rank,
                broad_rank=broad_rank,
                vix_level_rank=vix_rank,
            )
        )
        # Current observations are appended only after their ranks are fixed.
        front_history.append(front)
        broad_history.append(broad)
        vix_history.append(vix_level)
    return result


def decision_time(observation_date: date) -> datetime:
    local = datetime.combine(observation_date, time(9, 35), tzinfo=NEW_YORK)
    return local.astimezone(timezone.utc)


def _format(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("CVTR feature must be finite")
    return format(value, ".17g")


def build_events(
    rows: Iterable[SourceRow],
    *,
    mode: ClockMode = "primary",
    lookback_observations: int = 252,
    minimum_history: int = 126,
    lower_tail: float = 0.25,
    release_delay: int = 0,
) -> list[Event]:
    if not 0.0 < lower_tail < 0.5:
        raise ValueError("CVTR tail must be between zero and one-half")
    if release_delay not in {0, 1}:
        raise ValueError("CVTR release delay must be zero or one")
    source_rows = list(rows)
    features = build_features(
        source_rows,
        lookback_observations=lookback_observations,
        minimum_history=minimum_history,
    )
    result: list[Event] = []
    for feature in features:
        entry_index = feature.source_index + 1 + release_delay
        exit_index = entry_index + 1
        if exit_index >= len(source_rows):
            continue
        if feature.front_rank is None or feature.broad_rank is None or feature.vix_level_rank is None:
            continue
        if mode == "primary":
            score = 0.5 * (feature.front_rank + feature.broad_rank)
        elif mode == "front_only":
            score = feature.front_rank
        elif mode == "broad_only":
            score = feature.broad_rank
        elif mode == "vix_level":
            score = feature.vix_level_rank
        else:
            raise ValueError(f"unknown CVTR clock mode: {mode}")
        if score <= lower_tail:
            side = "LONG"
        elif score >= 1.0 - lower_tail:
            side = "SHORT"
        else:
            continue
        entry = decision_time(source_rows[entry_index].observation_date)
        exit_time = decision_time(source_rows[exit_index].observation_date)
        if exit_time <= entry:
            raise ValueError("CVTR exit must follow entry")
        result.append(
            Event(
                observation_date=feature.observation_date.isoformat(),
                signal_time=entry.isoformat(),
                entry_time=entry.isoformat(),
                exit_time=exit_time.isoformat(),
                side=side,
                clock_mode=mode if release_delay == 0 else "one_release_delay",
                front_slope=_format(feature.front_slope),
                broad_slope=_format(feature.broad_slope),
                front_rank=_format(feature.front_rank),
                broad_rank=_format(feature.broad_rank),
                vix_level_rank=_format(feature.vix_level_rank),
                score=_format(score),
            )
        )
    if len(result) > 1:
        for left, right in zip(result, result[1:]):
            if left.entry_time >= right.entry_time or left.exit_time > right.entry_time:
                raise ValueError("CVTR schedule overlaps or is not increasing")
    return result


def event_csv(events: Iterable[Event]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(EVENT_COLUMNS)
    for event in events:
        writer.writerow(tuple(getattr(event, column) for column in EVENT_COLUMNS))
    return output.getvalue().encode()


def write_events(path: str | Path, events: Iterable[Event]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as handle:
            handle.write(event_csv(events))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("primary", "front_only", "broad_only", "vix_level"),
        default="primary",
    )
    parser.add_argument("--release-delay", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    events = build_events(
        read_source(args.source),
        mode=cast(ClockMode, args.mode),
        release_delay=args.release_delay,
    )
    write_events(args.output, events)
    print(f"wrote {len(events)} source-only CVTR events to {args.output}")


if __name__ == "__main__":
    main()
