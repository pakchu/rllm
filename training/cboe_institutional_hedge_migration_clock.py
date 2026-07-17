"""Build outcome-blind Cboe Institutional Hedge Migration (CIHM-1) clocks."""
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

from training import build_cboe_option_flow_panel as source_builder


DEFAULT_SOURCE = (
    "data/cboe_option_flow_2020_2023/"
    "cboe_option_flow_2020-01-01_2023-12-31.csv.gz"
)
DEFAULT_OUTPUT = (
    "results/cboe_institutional_hedge_migration_"
    "preregistered_clock_2026-07-18.csv.gz"
)
SOURCE_COLUMNS = source_builder.PANEL_COLUMNS
EVENT_COLUMNS = (
    "observation_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "clock_mode",
    "institutional_gap",
    "vix_call_pressure",
    "index_share",
    "delta_institutional_gap",
    "delta_vix_call_pressure",
    "delta_index_share",
    "institutional_gap_rank",
    "vix_call_pressure_rank",
    "index_share_rank",
    "delta_institutional_gap_rank",
    "delta_vix_call_pressure_rank",
    "delta_index_share_rank",
    "score",
)
ClockMode = Literal[
    "primary",
    "institutional_gap_only",
    "vix_call_pressure_only",
    "index_share_only",
    "level_composite",
]
NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SourceRow:
    observation_date: date
    total_volume: int
    index_call_volume: int
    index_put_volume: int
    index_volume: int
    equity_call_volume: int
    equity_put_volume: int
    vix_call_volume: int
    vix_put_volume: int


@dataclass(frozen=True)
class FeatureRow:
    source_index: int
    observation_date: date
    institutional_gap: float
    vix_call_pressure: float
    index_share: float
    delta_institutional_gap: float | None
    delta_vix_call_pressure: float | None
    delta_index_share: float | None
    institutional_gap_rank: float | None
    vix_call_pressure_rank: float | None
    index_share_rank: float | None
    delta_institutional_gap_rank: float | None
    delta_vix_call_pressure_rank: float | None
    delta_index_share_rank: float | None


@dataclass(frozen=True)
class Event:
    observation_date: str
    signal_time: str
    entry_time: str
    exit_time: str
    side: str
    clock_mode: str
    institutional_gap: str
    vix_call_pressure: str
    index_share: str
    delta_institutional_gap: str
    delta_vix_call_pressure: str
    delta_index_share: str
    institutional_gap_rank: str
    vix_call_pressure_rank: str
    index_share_rank: str
    delta_institutional_gap_rank: str
    delta_vix_call_pressure_rank: str
    delta_index_share_rank: str
    score: str


def _nonnegative_integer(value: str, *, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be nonnegative")
    return parsed


def read_source(path: str | Path = DEFAULT_SOURCE) -> list[SourceRow]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError("Cboe option-flow panel schema changed")
        rows = [
            SourceRow(
                observation_date=date.fromisoformat(raw["observation_date"]),
                total_volume=_nonnegative_integer(
                    raw["total_volume"], field="total_volume"
                ),
                index_call_volume=_nonnegative_integer(
                    raw["index_call_volume"], field="index_call_volume"
                ),
                index_put_volume=_nonnegative_integer(
                    raw["index_put_volume"], field="index_put_volume"
                ),
                index_volume=_nonnegative_integer(
                    raw["index_volume"], field="index_volume"
                ),
                equity_call_volume=_nonnegative_integer(
                    raw["equity_call_volume"], field="equity_call_volume"
                ),
                equity_put_volume=_nonnegative_integer(
                    raw["equity_put_volume"], field="equity_put_volume"
                ),
                vix_call_volume=_nonnegative_integer(
                    raw["vix_call_volume"], field="vix_call_volume"
                ),
                vix_put_volume=_nonnegative_integer(
                    raw["vix_put_volume"], field="vix_put_volume"
                ),
            )
            for raw in reader
        ]
    if not rows:
        raise ValueError("Cboe option-flow panel is empty")
    dates = [row.observation_date for row in rows]
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("Cboe observation dates must be strictly increasing")
    if dates[-1] >= date(2024, 1, 1):
        raise ValueError("Cboe source escaped the frozen pre-2024 horizon")
    for row in rows:
        if min(
            row.total_volume,
            row.index_call_volume,
            row.index_put_volume,
            row.index_volume,
            row.equity_call_volume,
            row.equity_put_volume,
            row.vix_call_volume,
            row.vix_put_volume,
        ) <= 0:
            raise ValueError("CIHM log-ratio inputs must be positive")
        if row.index_volume > row.total_volume:
            raise ValueError("CIHM index volume exceeds total volume")
    return rows


def strict_prior_midrank(current: float, prior: list[float]) -> float:
    if not prior:
        raise ValueError("strict-prior rank needs history")
    less = sum(value < current for value in prior)
    equal = sum(value == current for value in prior)
    return (less + 0.5 * equal) / len(prior)


def _rank(current: float, history: list[float], lookback: int, minimum: int) -> float | None:
    prior = history[-lookback:]
    if len(prior) < minimum:
        return None
    return strict_prior_midrank(current, prior)


def _levels(row: SourceRow) -> tuple[float, float, float]:
    # Half-contract pseudocount is fixed only to make the transform defined if
    # a future source row contains a zero put or call count.
    institutional_gap = math.log(
        (row.index_put_volume + 0.5) / (row.index_call_volume + 0.5)
    ) - math.log(
        (row.equity_put_volume + 0.5) / (row.equity_call_volume + 0.5)
    )
    vix_call_pressure = math.log(
        (row.vix_call_volume + 0.5) / (row.vix_put_volume + 0.5)
    )
    index_share = math.log((row.index_volume + 1.0) / (row.total_volume + 1.0))
    return institutional_gap, vix_call_pressure, index_share


def build_features(
    rows: Iterable[SourceRow],
    *,
    lookback_observations: int = 252,
    minimum_history: int = 126,
) -> list[FeatureRow]:
    if lookback_observations < minimum_history or minimum_history < 3:
        raise ValueError("invalid CIHM rank history")
    source_rows = list(rows)
    level_histories: list[list[float]] = [[], [], []]
    delta_histories: list[list[float]] = [[], [], []]
    prior_levels: tuple[float, float, float] | None = None
    result: list[FeatureRow] = []
    for index, row in enumerate(source_rows):
        levels = _levels(row)
        deltas = (
            None
            if prior_levels is None
            else tuple(current - previous for current, previous in zip(levels, prior_levels))
        )
        level_ranks = tuple(
            _rank(value, history, lookback_observations, minimum_history)
            for value, history in zip(levels, level_histories)
        )
        if deltas is None:
            delta_ranks: tuple[float | None, ...] = (None, None, None)
        else:
            delta_ranks = tuple(
                _rank(value, history, lookback_observations, minimum_history)
                for value, history in zip(deltas, delta_histories)
            )
        result.append(
            FeatureRow(
                source_index=index,
                observation_date=row.observation_date,
                institutional_gap=levels[0],
                vix_call_pressure=levels[1],
                index_share=levels[2],
                delta_institutional_gap=None if deltas is None else deltas[0],
                delta_vix_call_pressure=None if deltas is None else deltas[1],
                delta_index_share=None if deltas is None else deltas[2],
                institutional_gap_rank=level_ranks[0],
                vix_call_pressure_rank=level_ranks[1],
                index_share_rank=level_ranks[2],
                delta_institutional_gap_rank=delta_ranks[0],
                delta_vix_call_pressure_rank=delta_ranks[1],
                delta_index_share_rank=delta_ranks[2],
            )
        )
        # Current values are appended only after every current rank is fixed.
        for history, value in zip(level_histories, levels):
            history.append(value)
        if deltas is not None:
            for history, value in zip(delta_histories, deltas):
                history.append(value)
        prior_levels = levels
    return result


def decision_time(observation_date: date) -> datetime:
    local = datetime.combine(observation_date, time(9, 35), tzinfo=NEW_YORK)
    return local.astimezone(timezone.utc)


def _format(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        raise ValueError("CIHM feature must be finite")
    return format(value, ".17g")


def _score(feature: FeatureRow, mode: ClockMode) -> float | None:
    delta_ranks = (
        feature.delta_institutional_gap_rank,
        feature.delta_vix_call_pressure_rank,
        feature.delta_index_share_rank,
    )
    level_ranks = (
        feature.institutional_gap_rank,
        feature.vix_call_pressure_rank,
        feature.index_share_rank,
    )
    if any(value is None for value in (*delta_ranks, *level_ranks)):
        return None
    if mode == "primary":
        return sum(cast(float, value) for value in delta_ranks) / 3.0
    if mode == "institutional_gap_only":
        return feature.delta_institutional_gap_rank
    if mode == "vix_call_pressure_only":
        return feature.delta_vix_call_pressure_rank
    if mode == "index_share_only":
        return feature.delta_index_share_rank
    if mode == "level_composite":
        return sum(cast(float, value) for value in level_ranks) / 3.0
    raise ValueError(f"unknown CIHM clock mode: {mode}")


def build_events(
    rows: Iterable[SourceRow],
    *,
    mode: ClockMode = "primary",
    lookback_observations: int = 252,
    minimum_history: int = 126,
    composite_threshold: float = 0.575,
    component_threshold: float = 0.70,
    release_delay: int = 0,
) -> list[Event]:
    if not 0.5 < composite_threshold < 1.0:
        raise ValueError("CIHM composite threshold must exceed one-half")
    if not 0.5 < component_threshold < 1.0:
        raise ValueError("CIHM component threshold must exceed one-half")
    if release_delay not in {0, 1, 7}:
        raise ValueError("CIHM release delay must be zero, one, or seven")
    source_rows = list(rows)
    features = build_features(
        source_rows,
        lookback_observations=lookback_observations,
        minimum_history=minimum_history,
    )
    result: list[Event] = []
    for feature in features:
        score = _score(feature, mode)
        threshold = (
            composite_threshold
            if mode in {"primary", "level_composite"}
            else component_threshold
        )
        if score is None or score < threshold:
            continue
        entry_index = feature.source_index + 1 + release_delay
        exit_index = entry_index + 1
        if exit_index >= len(source_rows):
            continue
        entry = decision_time(source_rows[entry_index].observation_date)
        exit_time = decision_time(source_rows[exit_index].observation_date)
        if exit_time <= entry:
            raise ValueError("CIHM exit must follow entry")
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
                institutional_gap=_format(feature.institutional_gap),
                vix_call_pressure=_format(feature.vix_call_pressure),
                index_share=_format(feature.index_share),
                delta_institutional_gap=_format(feature.delta_institutional_gap),
                delta_vix_call_pressure=_format(feature.delta_vix_call_pressure),
                delta_index_share=_format(feature.delta_index_share),
                institutional_gap_rank=_format(feature.institutional_gap_rank),
                vix_call_pressure_rank=_format(feature.vix_call_pressure_rank),
                index_share_rank=_format(feature.index_share_rank),
                delta_institutional_gap_rank=_format(
                    feature.delta_institutional_gap_rank
                ),
                delta_vix_call_pressure_rank=_format(
                    feature.delta_vix_call_pressure_rank
                ),
                delta_index_share_rank=_format(feature.delta_index_share_rank),
                score=_format(score),
            )
        )
    for left, right in zip(result, result[1:]):
        if left.entry_time >= right.entry_time or left.exit_time > right.entry_time:
            raise ValueError("CIHM schedule overlaps or is not increasing")
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
            "institutional_gap_only",
            "vix_call_pressure_only",
            "index_share_only",
            "level_composite",
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
    print(f"wrote {len(events)} outcome-blind CIHM events to {args.output}")


if __name__ == "__main__":
    main()
