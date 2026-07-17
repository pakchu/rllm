"""Build outcome-blind Cboe Tail-Hedge Disagreement (CTHD-1) clocks."""
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
    "data/cboe_tail_risk_2018_2023/"
    "cboe_tail_risk_2018-01-01_2023-12-31.csv.gz"
)
DEFAULT_OUTPUT = (
    "results/cboe_tail_hedge_disagreement_preregistered_clock_2026-07-18.csv.gz"
)
SOURCE_COLUMNS = ("observation_date", "SKEW_close", "VVIX_close", "VIX_close")
EVENT_COLUMNS = (
    "observation_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "clock_mode",
    "skew_level",
    "vvix_relative",
    "vix_level",
    "skew_rank",
    "vvix_relative_rank",
    "vix_level_rank",
    "hidden_pressure",
    "hidden_pressure_rank",
    "score",
)
ClockMode = Literal[
    "primary",
    "skew_only",
    "vvix_relative_only",
    "low_vix_only",
    "tail_pair_only",
]
NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SourceRow:
    observation_date: date
    skew_close: float
    vvix_close: float
    vix_close: float


@dataclass(frozen=True)
class FeatureRow:
    source_index: int
    observation_date: date
    skew_level: float
    vvix_relative: float
    vix_level: float
    skew_rank: float | None
    vvix_relative_rank: float | None
    vix_level_rank: float | None
    hidden_pressure: float | None
    hidden_pressure_rank: float | None


@dataclass(frozen=True)
class Event:
    observation_date: str
    signal_time: str
    entry_time: str
    exit_time: str
    side: str
    clock_mode: str
    skew_level: str
    vvix_relative: str
    vix_level: str
    skew_rank: str
    vvix_relative_rank: str
    vix_level_rank: str
    hidden_pressure: str
    hidden_pressure_rank: str
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
            raise ValueError("Cboe tail-risk panel schema changed")
        rows = [
            SourceRow(
                observation_date=date.fromisoformat(raw["observation_date"]),
                skew_close=_positive(raw["SKEW_close"], field="SKEW_close"),
                vvix_close=_positive(raw["VVIX_close"], field="VVIX_close"),
                vix_close=_positive(raw["VIX_close"], field="VIX_close"),
            )
            for raw in reader
        ]
    if not rows:
        raise ValueError("Cboe tail-risk panel is empty")
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
        raise ValueError("invalid CTHD rank history")
    source_rows = list(rows)
    skew_history: list[float] = []
    vvix_relative_history: list[float] = []
    vix_history: list[float] = []
    pressure_history: list[float] = []
    result: list[FeatureRow] = []
    for index, row in enumerate(source_rows):
        skew_level = math.log(row.skew_close / 100.0)
        vvix_relative = math.log(row.vvix_close / row.vix_close)
        vix_level = math.log(row.vix_close)
        skew_prior = skew_history[-lookback_observations:]
        vvix_prior = vvix_relative_history[-lookback_observations:]
        vix_prior = vix_history[-lookback_observations:]
        if min(len(skew_prior), len(vvix_prior), len(vix_prior)) >= minimum_history:
            skew_rank = strict_prior_midrank(skew_level, skew_prior)
            vvix_rank = strict_prior_midrank(vvix_relative, vvix_prior)
            vix_rank = strict_prior_midrank(vix_level, vix_prior)
            pressure = 0.5 * (skew_rank + vvix_rank) - vix_rank
            pressure_prior = pressure_history[-lookback_observations:]
            pressure_rank = (
                strict_prior_midrank(pressure, pressure_prior)
                if len(pressure_prior) >= minimum_history
                else None
            )
        else:
            skew_rank = None
            vvix_rank = None
            vix_rank = None
            pressure = None
            pressure_rank = None
        result.append(
            FeatureRow(
                source_index=index,
                observation_date=row.observation_date,
                skew_level=skew_level,
                vvix_relative=vvix_relative,
                vix_level=vix_level,
                skew_rank=skew_rank,
                vvix_relative_rank=vvix_rank,
                vix_level_rank=vix_rank,
                hidden_pressure=pressure,
                hidden_pressure_rank=pressure_rank,
            )
        )
        # Every current observation is appended only after its rank is fixed.
        skew_history.append(skew_level)
        vvix_relative_history.append(vvix_relative)
        vix_history.append(vix_level)
        if pressure is not None:
            pressure_history.append(pressure)
    return result


def decision_time(observation_date: date) -> datetime:
    local = datetime.combine(observation_date, time(9, 35), tzinfo=NEW_YORK)
    return local.astimezone(timezone.utc)


def _format(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        raise ValueError("CTHD feature must be finite")
    return format(value, ".17g")


def _score(feature: FeatureRow, mode: ClockMode) -> float | None:
    if mode == "primary":
        return feature.hidden_pressure_rank
    # Controls use the same doubly-warmed eligible universe as the primary.
    if (
        feature.hidden_pressure_rank is None
        or feature.skew_rank is None
        or feature.vvix_relative_rank is None
        or feature.vix_level_rank is None
    ):
        return None
    if mode == "skew_only":
        return feature.skew_rank
    if mode == "vvix_relative_only":
        return feature.vvix_relative_rank
    if mode == "low_vix_only":
        return 1.0 - feature.vix_level_rank
    if mode == "tail_pair_only":
        return 0.5 * (feature.skew_rank + feature.vvix_relative_rank)
    raise ValueError(f"unknown CTHD clock mode: {mode}")


def build_events(
    rows: Iterable[SourceRow],
    *,
    mode: ClockMode = "primary",
    lookback_observations: int = 252,
    minimum_history: int = 126,
    upper_tail: float = 0.225,
    release_delay: int = 0,
) -> list[Event]:
    if not 0.0 < upper_tail < 0.5:
        raise ValueError("CTHD tail must be between zero and one-half")
    if release_delay not in {0, 1, 7}:
        raise ValueError("CTHD release delay must be zero, one, or seven")
    source_rows = list(rows)
    features = build_features(
        source_rows,
        lookback_observations=lookback_observations,
        minimum_history=minimum_history,
    )
    result: list[Event] = []
    for feature in features:
        score = _score(feature, mode)
        if score is None or score < 1.0 - upper_tail:
            continue
        entry_index = feature.source_index + 1 + release_delay
        exit_index = entry_index + 1
        if exit_index >= len(source_rows):
            continue
        entry = decision_time(source_rows[entry_index].observation_date)
        exit_time = decision_time(source_rows[exit_index].observation_date)
        if exit_time <= entry:
            raise ValueError("CTHD exit must follow entry")
        if release_delay == 1:
            clock_mode = "one_release_delay"
        elif release_delay == 7:
            clock_mode = "seven_release_placebo"
        else:
            clock_mode = mode
        result.append(
            Event(
                observation_date=feature.observation_date.isoformat(),
                signal_time=entry.isoformat(),
                entry_time=entry.isoformat(),
                exit_time=exit_time.isoformat(),
                side="SHORT",
                clock_mode=clock_mode,
                skew_level=_format(feature.skew_level),
                vvix_relative=_format(feature.vvix_relative),
                vix_level=_format(feature.vix_level),
                skew_rank=_format(feature.skew_rank),
                vvix_relative_rank=_format(feature.vvix_relative_rank),
                vix_level_rank=_format(feature.vix_level_rank),
                hidden_pressure=_format(feature.hidden_pressure),
                hidden_pressure_rank=_format(feature.hidden_pressure_rank),
                score=_format(score),
            )
        )
    for left, right in zip(result, result[1:]):
        if left.entry_time >= right.entry_time or left.exit_time > right.entry_time:
            raise ValueError("CTHD schedule overlaps or is not increasing")
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
        choices=(
            "primary",
            "skew_only",
            "vvix_relative_only",
            "low_vix_only",
            "tail_pair_only",
        ),
        default="primary",
    )
    parser.add_argument("--release-delay", type=int, choices=(0, 1, 7), default=0)
    args = parser.parse_args()
    mode = cast(ClockMode, args.mode)
    events = build_events(
        read_source(args.source), mode=mode, release_delay=args.release_delay
    )
    write_events(args.output, events)
    print(f"wrote {len(events)} outcome-blind CTHD events to {args.output}")


if __name__ == "__main__":
    main()
