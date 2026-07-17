"""Exact source-only clock for Federal Liquidity Component Concordance.

The clock combines three weak, public-liquidity contributions from each H.4.1
release: Federal Reserve asset expansion, Treasury General Account release,
and reverse-repo release.  It never opens a crypto market or outcome field.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


SOURCE_PATH = (
    "data/federal_reserve_h41_net_liquidity_2018_2023/"
    "federal_reserve_h41_net_liquidity_2018-01-04_2023-12-28.csv.gz"
)
DEFAULT_OUTPUT = (
    "results/federal_liquidity_component_concordance_"
    "preregistered_clock_2026-07-17.csv.gz"
)
PRIOR_LOOKBACK = 104
MINIMUM_COMPONENT_BREADTH = 2
ENTRY_DELAY = timedelta(minutes=5)
HOLD = timedelta(days=5)
RESEARCH_END = datetime(2024, 1, 1, tzinfo=timezone.utc)

CLOCK_NAMES = (
    "primary",
    "net_only",
    "component_concordance_only",
    "direction_flip",
    "one_release_delay",
    "random_side",
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    horizon_releases: int
    lower_rank_numerator: int
    upper_rank_numerator: int


CANDIDATE_SPECS = (
    CandidateSpec("FLCC-H4-Q60", 4, 83, 125),
    CandidateSpec("FLCC-H4-Q65", 4, 72, 136),
    CandidateSpec("FLCC-H8-Q60", 8, 83, 125),
    CandidateSpec("FLCC-H8-Q65", 8, 72, 136),
)


@dataclass(frozen=True)
class SourceRow:
    release_date: str
    observation_date: str
    available_at_utc: str
    total_assets_usd_millions: int
    treasury_general_account_usd_millions: int
    reverse_repurchase_agreements_usd_millions: int
    net_liquidity_usd_millions: int


@dataclass(frozen=True)
class FeatureRow:
    source_index: int
    release_date: str
    available_at_utc: str
    net_rank_numerator: int
    asset_rank_numerator: int
    tga_release_rank_numerator: int
    rrp_release_rank_numerator: int
    component_breadth: int
    component_tail_breadth: int
    side: int


@dataclass(frozen=True)
class Event:
    candidate_id: str
    clock_name: str
    feature_release_date: str
    signal_release_date: str
    signal_time: str
    entry_time: str
    exit_time: str
    side: int
    horizon_releases: int
    lower_rank_numerator: int
    upper_rank_numerator: int
    prior_lookback: int
    net_rank_numerator: int
    asset_rank_numerator: int
    tga_release_rank_numerator: int
    rrp_release_rank_numerator: int
    component_breadth: int
    component_tail_breadth: int


EVENT_COLUMNS = tuple(Event.__dataclass_fields__)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp must be timezone-aware: {value!r}")
    return parsed.astimezone(timezone.utc)


def _parse_int(value: str, *, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative, got {parsed}")
    return parsed


def read_source(path: str | Path = SOURCE_PATH) -> list[SourceRow]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[SourceRow] = []
    for raw in raw_rows:
        row = SourceRow(
            release_date=raw["release_date"],
            observation_date=raw["observation_date"],
            available_at_utc=raw["available_at_utc"],
            total_assets_usd_millions=_parse_int(
                raw["total_assets_usd_millions"],
                field="total_assets_usd_millions",
            ),
            treasury_general_account_usd_millions=_parse_int(
                raw["treasury_general_account_usd_millions"],
                field="treasury_general_account_usd_millions",
            ),
            reverse_repurchase_agreements_usd_millions=_parse_int(
                raw["reverse_repurchase_agreements_usd_millions"],
                field="reverse_repurchase_agreements_usd_millions",
            ),
            net_liquidity_usd_millions=_parse_int(
                raw["net_liquidity_usd_millions"],
                field="net_liquidity_usd_millions",
            ),
        )
        expected_net = (
            row.total_assets_usd_millions
            - row.treasury_general_account_usd_millions
            - row.reverse_repurchase_agreements_usd_millions
        )
        if expected_net != row.net_liquidity_usd_millions:
            raise ValueError(f"net-liquidity identity failed on {row.release_date}")
        released = datetime.fromisoformat(row.release_date).date()
        observed = datetime.fromisoformat(row.observation_date).date()
        available = _parse_utc(row.available_at_utc)
        if not observed < released:
            raise ValueError(f"noncausal H.4.1 observation on {row.release_date}")
        if available.date() != released:
            raise ValueError(f"H.4.1 availability date differs on {row.release_date}")
        if released.year >= 2024:
            raise ValueError("2024+ H.4.1 source is sealed")
        rows.append(row)
    release_dates = [row.release_date for row in rows]
    if release_dates != sorted(release_dates):
        raise ValueError("H.4.1 source is not sorted by release date")
    if len(set(release_dates)) != len(release_dates):
        raise ValueError("H.4.1 source contains duplicate releases")
    return rows


def _midrank_numerator(current: int, prior: Iterable[int]) -> int:
    exact_prior = list(prior)
    if len(exact_prior) != PRIOR_LOOKBACK:
        raise ValueError(
            f"midrank requires {PRIOR_LOOKBACK} prior values, got {len(exact_prior)}"
        )
    return 2 * sum(value < current for value in exact_prior) + sum(
        value == current for value in exact_prior
    )


def _impulses(rows: list[SourceRow], horizon: int) -> dict[str, list[int | None]]:
    levels = {
        "asset": [row.total_assets_usd_millions for row in rows],
        "tga_release": [-row.treasury_general_account_usd_millions for row in rows],
        "rrp_release": [
            -row.reverse_repurchase_agreements_usd_millions for row in rows
        ],
        "net": [row.net_liquidity_usd_millions for row in rows],
    }
    output: dict[str, list[int | None]] = {}
    for name, values in levels.items():
        output[name] = [
            None if index < horizon else values[index] - values[index - horizon]
            for index in range(len(values))
        ]
    return output


def build_features(rows: list[SourceRow], spec: CandidateSpec) -> list[FeatureRow]:
    impulses = _impulses(rows, spec.horizon_releases)
    output: list[FeatureRow] = []
    first_index = spec.horizon_releases + PRIOR_LOOKBACK
    for index in range(first_index, len(rows)):
        ranks: dict[str, int] = {}
        for name, values in impulses.items():
            current = values[index]
            prior = values[index - PRIOR_LOOKBACK : index]
            if current is None or any(value is None for value in prior):
                raise ValueError("H.4.1 feature warm-up contract failed")
            ranks[name] = _midrank_numerator(
                current,
                (int(value) for value in prior if value is not None),
            )
        net_centered = ranks["net"] - PRIOR_LOOKBACK
        side = 1 if net_centered > 0 else -1 if net_centered < 0 else 0
        component_ranks = (
            ranks["asset"],
            ranks["tga_release"],
            ranks["rrp_release"],
        )
        component_breadth = sum(
            (rank - PRIOR_LOOKBACK) * side > 0 for rank in component_ranks
        )
        component_tail_breadth = sum(
            (side > 0 and rank >= spec.upper_rank_numerator)
            or (side < 0 and rank <= spec.lower_rank_numerator)
            for rank in component_ranks
        )
        row = rows[index]
        output.append(
            FeatureRow(
                source_index=index,
                release_date=row.release_date,
                available_at_utc=row.available_at_utc,
                net_rank_numerator=ranks["net"],
                asset_rank_numerator=ranks["asset"],
                tga_release_rank_numerator=ranks["tga_release"],
                rrp_release_rank_numerator=ranks["rrp_release"],
                component_breadth=component_breadth,
                component_tail_breadth=component_tail_breadth,
                side=side,
            )
        )
    return output


def _net_tail(feature: FeatureRow, spec: CandidateSpec) -> bool:
    return (
        feature.net_rank_numerator <= spec.lower_rank_numerator
        or feature.net_rank_numerator >= spec.upper_rank_numerator
    )


def _event(
    spec: CandidateSpec,
    clock_name: str,
    feature: FeatureRow,
    signal_row: SourceRow,
    *,
    side: int,
) -> Event:
    if side not in (-1, 1):
        raise ValueError(f"event side must be -1 or 1, got {side}")
    signal_time = _parse_utc(signal_row.available_at_utc)
    entry_time = signal_time + ENTRY_DELAY
    exit_time = entry_time + HOLD
    return Event(
        candidate_id=spec.candidate_id,
        clock_name=clock_name,
        feature_release_date=feature.release_date,
        signal_release_date=signal_row.release_date,
        signal_time=signal_time.isoformat(),
        entry_time=entry_time.isoformat(),
        exit_time=exit_time.isoformat(),
        side=side,
        horizon_releases=spec.horizon_releases,
        lower_rank_numerator=spec.lower_rank_numerator,
        upper_rank_numerator=spec.upper_rank_numerator,
        prior_lookback=PRIOR_LOOKBACK,
        net_rank_numerator=feature.net_rank_numerator,
        asset_rank_numerator=feature.asset_rank_numerator,
        tga_release_rank_numerator=feature.tga_release_rank_numerator,
        rrp_release_rank_numerator=feature.rrp_release_rank_numerator,
        component_breadth=feature.component_breadth,
        component_tail_breadth=feature.component_tail_breadth,
    )


def _random_side(spec: CandidateSpec, feature: FeatureRow) -> int:
    digest = hashlib.sha256(
        f"FLCC-RANDOM-SIDE-V1|{spec.candidate_id}|{feature.release_date}".encode()
    ).digest()
    return 1 if digest[0] & 1 else -1


def build_raw_events(
    rows: list[SourceRow],
    spec: CandidateSpec,
) -> dict[str, list[Event]]:
    features = build_features(rows, spec)
    output = {name: [] for name in CLOCK_NAMES}
    for feature in features:
        if feature.side == 0:
            continue
        source_row = rows[feature.source_index]
        primary_active = (
            _net_tail(feature, spec)
            and feature.component_breadth >= MINIMUM_COMPONENT_BREADTH
        )
        if _net_tail(feature, spec):
            output["net_only"].append(
                _event(spec, "net_only", feature, source_row, side=feature.side)
            )
        if feature.component_tail_breadth >= MINIMUM_COMPONENT_BREADTH:
            output["component_concordance_only"].append(
                _event(
                    spec,
                    "component_concordance_only",
                    feature,
                    source_row,
                    side=feature.side,
                )
            )
        if not primary_active:
            continue
        output["primary"].append(
            _event(spec, "primary", feature, source_row, side=feature.side)
        )
        output["direction_flip"].append(
            _event(spec, "direction_flip", feature, source_row, side=-feature.side)
        )
        output["random_side"].append(
            _event(
                spec,
                "random_side",
                feature,
                source_row,
                side=_random_side(spec, feature),
            )
        )
        delayed_index = feature.source_index + 1
        if delayed_index < len(rows):
            output["one_release_delay"].append(
                _event(
                    spec,
                    "one_release_delay",
                    feature,
                    rows[delayed_index],
                    side=feature.side,
                )
            )
    return output


def reserve_nonoverlap(events: Iterable[Event]) -> list[Event]:
    reserved: list[Event] = []
    active_until: datetime | None = None
    for event in sorted(events, key=lambda item: _parse_utc(item.entry_time)):
        entry = _parse_utc(event.entry_time)
        exit_time = _parse_utc(event.exit_time)
        if exit_time >= RESEARCH_END:
            continue
        if active_until is not None and entry < active_until:
            continue
        reserved.append(event)
        active_until = exit_time
    return reserved


def build_all_events(rows: list[SourceRow]) -> list[Event]:
    output: list[Event] = []
    for spec in CANDIDATE_SPECS:
        clocks = build_raw_events(rows, spec)
        for clock_name in CLOCK_NAMES:
            output.extend(reserve_nonoverlap(clocks[clock_name]))
    return sorted(
        output,
        key=lambda event: (event.candidate_id, event.clock_name, event.entry_time),
    )


def write_event_ledger(path: str | Path, events: Iterable[Event]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(EVENT_COLUMNS)
                for event in events:
                    writer.writerow(
                        [getattr(event, column) for column in EVENT_COLUMNS]
                    )
    return hashlib.sha256(target.read_bytes()).hexdigest()


def read_event_ledger(path: str | Path = DEFAULT_OUTPUT) -> list[Event]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Event(
            candidate_id=row["candidate_id"],
            clock_name=row["clock_name"],
            feature_release_date=row["feature_release_date"],
            signal_release_date=row["signal_release_date"],
            signal_time=row["signal_time"],
            entry_time=row["entry_time"],
            exit_time=row["exit_time"],
            side=int(row["side"]),
            horizon_releases=int(row["horizon_releases"]),
            lower_rank_numerator=int(row["lower_rank_numerator"]),
            upper_rank_numerator=int(row["upper_rank_numerator"]),
            prior_lookback=int(row["prior_lookback"]),
            net_rank_numerator=int(row["net_rank_numerator"]),
            asset_rank_numerator=int(row["asset_rank_numerator"]),
            tga_release_rank_numerator=int(row["tga_release_rank_numerator"]),
            rrp_release_rank_numerator=int(row["rrp_release_rank_numerator"]),
            component_breadth=int(row["component_breadth"]),
            component_tail_breadth=int(row["component_tail_breadth"]),
        )
        for row in rows
    ]


def main() -> None:
    events = build_all_events(read_source())
    digest = write_event_ledger(DEFAULT_OUTPUT, events)
    print(f"events={len(events)} sha256={digest} output={DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
