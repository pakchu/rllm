"""Build outcome-blind RMSR-72 source-support, control, and novelty clocks."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from training import build_ofr_repo_venue_fragmentation_consensus_support as rvfc_support
from training import preregister_ofr_repo_mix_shock_resolution_race as prereg


PROTOCOL_VERSION = "ofr_repo_mix_shock_resolution_race_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_ofr_repo_mix_shock_resolution_race_support.py")
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "3a85ac8fa41aa4caacb6ff4875ec4e19d01ec0450f8959f84cb16787a5cc3f45"
)
PREREGISTRATION_MANIFEST_HASH = (
    "a9ad2e858e9b037e459a1f5fe2cfba1af5a921a35ab100a4d1eca027807baa53"
)
DEFAULT_CLOCK = Path(
    "results/ofr_repo_mix_shock_resolution_race_clocks_2026-07-23.csv.gz"
)
DEFAULT_REPORT = Path(
    "results/ofr_repo_mix_shock_resolution_race_support_2026-07-23.json"
)

UTC = timezone.utc
BAR = timedelta(minutes=5)
HOLD = timedelta(hours=72)
LOOKBACK = 252
PRIMARY_WINDOW = 20
START_DATE = date(2019, 1, 1)
END_DATE = date(2023, 12, 31)
FEED_FLOOR = datetime(2020, 9, 10, tzinfo=UTC)
TRAIN_START = datetime(2021, 1, 1, tzinfo=UTC)
TRAIN_END = datetime(2023, 1, 1, tzinfo=UTC)
SELECTION_START = TRAIN_END
SELECTION_END = datetime(2024, 1, 1, tzinfo=UTC)

SOURCE_CONTROLS = (
    "mix_transition_only",
    "rate_transition_only",
    "price_confirmation_only",
    "quantity_absorption_only",
    "reverse_race",
    "five_date_window",
    "forty_date_window",
    "one_complete_date_stale",
    "five_complete_date_stale",
    "year_rate_permutation",
    "same_date_alignment",
)
ECONOMIC_CONTROLS = (
    "exact_direction_flip",
    "deterministic_random_side",
    "constant_long",
    "constant_short",
)
CONTROL_NAMES = ("primary", *SOURCE_CONTROLS, *ECONOMIC_CONTROLS)

SOURCE_COLUMNS = (
    "mnemonic",
    "observation_date",
    "available_at_utc",
    "value",
    "disclosure_edit",
    "segment",
    "measure",
    "subset",
    "series_name",
)
CLOCK_COLUMNS = (
    "control",
    "precursor_observation_date",
    "terminal_observation_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "split",
    "side",
    "precursor_polarity",
    "precursor_lag_state",
    "terminal_type",
    "terminal_age_dates",
    "u_mix_disagreement",
    "u_rate_disagreement",
    "dominant_collateral_spread_venue",
)


@dataclass(frozen=True)
class SourceRow:
    mnemonic: str
    observation_date: date
    available_at: datetime
    value: Fraction | None
    disclosure_edit: str = ""


@dataclass(frozen=True)
class FeatureRow:
    observation_date: date
    available_at: datetime
    epoch: int
    decision_allowed: bool
    components: Mapping[str, Fraction]
    dominant_collateral_spread_venue: str


@dataclass(frozen=True)
class RankRow:
    feature: FeatureRow
    units: Mapping[str, Fraction]


@dataclass(frozen=True)
class StateRow:
    rank: RankRow
    epoch: int
    mix_state: int
    rate_state: int


@dataclass(frozen=True)
class RaceCandidate:
    precursor_observation_date: date
    terminal_observation_date: date
    signal_time: datetime
    side: int
    precursor_polarity: int
    precursor_lag_state: int
    terminal_type: str
    terminal_age_dates: int
    units: Mapping[str, Fraction]
    dominant_collateral_spread_venue: str


@dataclass(frozen=True)
class Scheduled:
    control: str
    precursor_observation_date: date
    terminal_observation_date: date
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    split: str
    side: int
    precursor_polarity: int
    precursor_lag_state: int
    terminal_type: str
    terminal_age_dates: int
    units: Mapping[str, Fraction]
    dominant_collateral_spread_venue: str


@dataclass(frozen=True)
class ComparatorEvent:
    entry_time: datetime
    exit_time: datetime
    side: int


class ComparatorValidationError(RuntimeError):
    def __init__(self, message: str, *, rows_read: int) -> None:
        super().__init__(message)
        self.rows_read = rows_read


@dataclass(frozen=True)
class SourceAudit:
    normalized_rows_read: int
    required_rows_read: int
    source_dates_seen: int
    valid_feature_dates: int
    invalid_missing_or_null_dates: int
    invalid_materiality_dates: int
    equal_availability_rows_suppressed: int


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("RMSR support path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RMSR support path must remain repository-relative") from exc
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("RMSR timestamp must be text")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"invalid RMSR timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError("RMSR timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _fraction(value: Any, label: str, *, optional: bool = False) -> Fraction | None:
    if optional and value == "":
        return None
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be numeric")
    try:
        decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RuntimeError(f"{label} must be decimal") from exc
    if not decimal.is_finite():
        raise RuntimeError(f"{label} must be finite")
    return Fraction(decimal)


def _side(value: Any) -> int:
    text = str(value).strip().upper()
    if text in {"1", "+1", "LONG"}:
        return 1
    if text in {"-1", "SHORT"}:
        return -1
    raise RuntimeError(f"invalid RMSR comparator side: {value}")


def _expected_availability(day: date) -> datetime:
    delayed = datetime.combine(day + timedelta(days=8), time(), UTC)
    return max(delayed, FEED_FLOOR)


def _load_registration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("RMSR preregistration file hash mismatch")
    payload = json.loads(_repository_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_preregistration(payload, verify_sources=True)
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("RMSR preregistration manifest hash mismatch")
    return payload


def load_source() -> tuple[dict[date, dict[str, SourceRow]], int, int]:
    if sha256_file(prereg.OBSERVATIONS) != prereg.OBSERVATIONS_SHA256:
        raise RuntimeError("RMSR source observation hash mismatch")
    required = set(prereg.REQUIRED_SERIES)
    by_date: dict[date, dict[str, SourceRow]] = defaultdict(dict)
    all_dates: set[date] = set()
    normalized_rows = required_rows = 0
    with gzip.open(_repository_path(prereg.OBSERVATIONS), "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise RuntimeError("RMSR source columns changed")
        for raw in reader:
            normalized_rows += 1
            try:
                observation_day = date.fromisoformat(raw["observation_date"])
            except ValueError as exc:
                raise RuntimeError("RMSR observation date changed") from exc
            if not START_DATE <= observation_day <= END_DATE:
                raise RuntimeError("RMSR source date escaped frozen window")
            all_dates.add(observation_day)
            mnemonic = raw["mnemonic"]
            if mnemonic not in required:
                continue
            required_rows += 1
            available_at = _timestamp(raw["available_at_utc"])
            if available_at != _expected_availability(observation_day):
                raise RuntimeError("RMSR source availability changed")
            if mnemonic in by_date[observation_day]:
                raise RuntimeError("duplicate RMSR required source row")
            by_date[observation_day][mnemonic] = SourceRow(
                mnemonic=mnemonic,
                observation_date=observation_day,
                available_at=available_at,
                value=_fraction(raw["value"], f"{mnemonic} value", optional=True),
                disclosure_edit=raw["disclosure_edit"].strip(),
            )
    if normalized_rows != 77_369:
        raise RuntimeError("RMSR normalized source row count changed")
    for observation_day in all_dates:
        by_date.setdefault(observation_day, {})
    if not by_date:
        raise RuntimeError("RMSR required source is empty")
    return dict(by_date), normalized_rows, required_rows


def _required_value(rows: Mapping[str, SourceRow], mnemonic: str) -> Fraction:
    row = rows[mnemonic]
    if row.value is None:
        raise RuntimeError("required RMSR value is null")
    return row.value


def _dominant(values: Mapping[str, Fraction]) -> str:
    maximum = max(values.values())
    winners = sorted(name for name, value in values.items() if value == maximum)
    return winners[0] if len(winners) == 1 else "TIE:" + "+".join(winners)


def build_features(
    by_date: Mapping[date, Mapping[str, SourceRow]],
    *,
    normalized_rows_read: int = 0,
    required_rows_read: int = 0,
) -> tuple[list[FeatureRow], SourceAudit]:
    required = set(prereg.REQUIRED_SERIES)
    features: list[FeatureRow] = []
    epoch = 0
    invalid_missing_or_null = 0
    invalid_materiality = 0
    for observation_day in sorted(by_date):
        rows = by_date[observation_day]
        if set(rows) != required or any(
            row.value is None or row.disclosure_edit for row in rows.values()
        ):
            epoch += 1
            invalid_missing_or_null += 1
            continue
        value = lambda mnemonic: _required_value(rows, mnemonic)
        gcf_ag = value("REPO-GCF_TV_AG-P")
        gcf_t = value("REPO-GCF_TV_T-P")
        tri_ag = value("REPO-TRIV1_TV_AG-P")
        tri_t = value("REPO-TRIV1_TV_T-P")
        gcf_total = gcf_ag + gcf_t
        tri_total = tri_ag + tri_t
        material = (
            gcf_total > 0
            and tri_total > 0
            and gcf_ag * 20 >= gcf_total
            and gcf_t * 20 >= gcf_total
            and tri_ag * 20 >= tri_total
            and tri_t * 20 >= tri_total
        )
        if not material:
            epoch += 1
            invalid_materiality += 1
            continue
        spreads = {
            "GCF": abs(value("REPO-GCF_AR_AG-P") - value("REPO-GCF_AR_T-P")),
            "TRIV1": abs(
                value("REPO-TRIV1_AR_AG-P") - value("REPO-TRIV1_AR_T-P")
            ),
        }
        components = {
            "mix_disagreement": abs(gcf_ag / gcf_total - tri_ag / tri_total),
            "rate_disagreement": sum(spreads.values(), Fraction()) / 2,
        }
        features.append(
            FeatureRow(
                observation_date=observation_day,
                available_at=max(row.available_at for row in rows.values()),
                epoch=epoch,
                decision_allowed=False,
                components=components,
                dominant_collateral_spread_venue=_dominant(spreads),
            )
        )
    latest_by_availability: dict[datetime, date] = {}
    for row in features:
        latest_by_availability[row.available_at] = max(
            row.observation_date,
            latest_by_availability.get(row.available_at, row.observation_date),
        )
    features = [
        replace(
            row,
            decision_allowed=(
                latest_by_availability[row.available_at] == row.observation_date
            ),
        )
        for row in features
    ]
    audit = SourceAudit(
        normalized_rows_read=normalized_rows_read,
        required_rows_read=required_rows_read,
        source_dates_seen=len(by_date),
        valid_feature_dates=len(features),
        invalid_missing_or_null_dates=invalid_missing_or_null,
        invalid_materiality_dates=invalid_materiality,
        equal_availability_rows_suppressed=sum(
            not row.decision_allowed for row in features
        ),
    )
    return features, audit


def midrank_unit(current: Fraction, prior: Sequence[Fraction]) -> Fraction:
    if len(prior) != LOOKBACK:
        raise RuntimeError("RMSR midrank history length changed")
    less = sum(value < current for value in prior)
    equal = sum(value == current for value in prior)
    return Fraction(2 * less + equal - LOOKBACK, LOOKBACK)


def build_rank_rows(features: Sequence[FeatureRow]) -> list[RankRow]:
    histories = {name: deque(maxlen=LOOKBACK) for name in prereg.COMPONENTS}
    output: list[RankRow] = []
    batches: dict[datetime, list[FeatureRow]] = defaultdict(list)
    for row in features:
        batches[row.available_at].append(row)
    for available_at in sorted(batches):
        batch = sorted(batches[available_at], key=lambda row: row.observation_date)
        if all(len(history) == LOOKBACK for history in histories.values()):
            prior = {name: tuple(history) for name, history in histories.items()}
            output.extend(
                RankRow(
                    feature=row,
                    units={
                        name: midrank_unit(row.components[name], prior[name])
                        for name in prereg.COMPONENTS
                    },
                )
                for row in batch
            )
        for row in batch:
            for name in prereg.COMPONENTS:
                histories[name].append(row.components[name])
    return output


def component_state(unit: Fraction) -> int:
    if unit >= Fraction(1, 2):
        return 1
    if unit <= Fraction(-1, 2):
        return -1
    return 0


def build_state_rows(rank_rows: Sequence[RankRow]) -> list[StateRow]:
    return [
        StateRow(
            rank=row,
            epoch=row.feature.epoch,
            mix_state=component_state(row.units["mix_disagreement"]),
            rate_state=component_state(row.units["rate_disagreement"]),
        )
        for row in rank_rows
        if row.feature.decision_allowed
    ]


def _state(row: StateRow, name: str) -> int:
    if name == "mix":
        return row.mix_state
    if name == "rate":
        return row.rate_state
    raise RuntimeError(f"unknown RMSR state component: {name}")


def derive_race_candidates(
    rows: Sequence[StateRow],
    *,
    lead: str = "mix",
    lag: str = "rate",
    window: int = PRIMARY_WINDOW,
) -> tuple[list[RaceCandidate], dict[str, int]]:
    if lead == lag or {lead, lag} != {"mix", "rate"}:
        raise RuntimeError("RMSR race requires distinct mix and rate components")
    if window <= 0:
        raise RuntimeError("RMSR race window must be positive")
    audit = {
        "lead_extreme_transitions": 0,
        "armed": 0,
        "already_priced": 0,
        "price_confirmation": 0,
        "quantity_absorption": 0,
        "ambiguous_same_date": 0,
        "timeouts": 0,
        "continuity_cancellations": 0,
    }
    output: list[RaceCandidate] = []
    previous: StateRow | None = None
    active: dict[str, Any] | None = None
    for current in rows:
        if previous is None or current.epoch != previous.epoch:
            if active is not None:
                audit["continuity_cancellations"] += 1
            active = None
            previous = current
            continue

        terminal_consumed_date = False
        if active is not None:
            active["age"] += 1
            polarity = int(active["polarity"])
            confirmation = (
                _state(current, lag) == polarity
                and _state(previous, lag) != polarity
            )
            lead_exit = (
                _state(current, lead) != polarity
                and _state(previous, lead) == polarity
            )
            if confirmation and lead_exit:
                audit["ambiguous_same_date"] += 1
                active = None
                terminal_consumed_date = True
            elif confirmation:
                audit["price_confirmation"] += 1
                output.append(
                    RaceCandidate(
                        precursor_observation_date=active["precursor_date"],
                        terminal_observation_date=current.rank.feature.observation_date,
                        signal_time=current.rank.feature.available_at,
                        side=-polarity,
                        precursor_polarity=polarity,
                        precursor_lag_state=active["precursor_lag_state"],
                        terminal_type="PRICE_CONFIRMATION",
                        terminal_age_dates=active["age"],
                        units=current.rank.units,
                        dominant_collateral_spread_venue=(
                            current.rank.feature.dominant_collateral_spread_venue
                        ),
                    )
                )
                active = None
                terminal_consumed_date = True
            elif lead_exit:
                audit["quantity_absorption"] += 1
                output.append(
                    RaceCandidate(
                        precursor_observation_date=active["precursor_date"],
                        terminal_observation_date=current.rank.feature.observation_date,
                        signal_time=current.rank.feature.available_at,
                        side=polarity,
                        precursor_polarity=polarity,
                        precursor_lag_state=active["precursor_lag_state"],
                        terminal_type="QUANTITY_ABSORPTION",
                        terminal_age_dates=active["age"],
                        units=current.rank.units,
                        dominant_collateral_spread_venue=(
                            current.rank.feature.dominant_collateral_spread_venue
                        ),
                    )
                )
                active = None
                terminal_consumed_date = True
            elif active["age"] >= window:
                audit["timeouts"] += 1
                active = None
                terminal_consumed_date = True

        if active is None and not terminal_consumed_date:
            polarity = _state(current, lead)
            lead_transition = polarity in {-1, 1} and _state(previous, lead) != polarity
            if lead_transition:
                audit["lead_extreme_transitions"] += 1
                lag_state = _state(current, lag)
                if lag_state == polarity:
                    audit["already_priced"] += 1
                else:
                    audit["armed"] += 1
                    active = {
                        "precursor_date": current.rank.feature.observation_date,
                        "polarity": polarity,
                        "precursor_lag_state": lag_state,
                        "age": 0,
                    }
        previous = current
    if active is not None:
        audit["continuity_cancellations"] += 1
    return output, audit


def _transition_candidates(
    rows: Sequence[StateRow], *, component: str
) -> list[RaceCandidate]:
    output: list[RaceCandidate] = []
    previous: StateRow | None = None
    for current in rows:
        if previous is None or current.epoch != previous.epoch:
            previous = current
            continue
        polarity = _state(current, component)
        if polarity not in {-1, 1} or _state(previous, component) == polarity:
            previous = current
            continue
        lag_state = current.rate_state if component == "mix" else current.mix_state
        if component == "mix" and lag_state == polarity:
            previous = current
            continue
        output.append(
            RaceCandidate(
                precursor_observation_date=current.rank.feature.observation_date,
                terminal_observation_date=current.rank.feature.observation_date,
                signal_time=current.rank.feature.available_at,
                side=-polarity,
                precursor_polarity=polarity,
                precursor_lag_state=lag_state,
                terminal_type=f"{component.upper()}_TRANSITION",
                terminal_age_dates=0,
                units=current.rank.units,
                dominant_collateral_spread_venue=(
                    current.rank.feature.dominant_collateral_spread_venue
                ),
            )
        )
        previous = current
    return output


def _same_date_alignment_candidates(
    rows: Sequence[StateRow],
) -> list[RaceCandidate]:
    output: list[RaceCandidate] = []
    previous: StateRow | None = None
    for current in rows:
        if previous is None or current.epoch != previous.epoch:
            previous = current
            continue
        polarity = current.mix_state
        aligned = (
            polarity in {-1, 1}
            and previous.mix_state != polarity
            and current.rate_state == polarity
            and previous.rate_state != polarity
        )
        if aligned:
            output.append(
                RaceCandidate(
                    precursor_observation_date=current.rank.feature.observation_date,
                    terminal_observation_date=current.rank.feature.observation_date,
                    signal_time=current.rank.feature.available_at,
                    side=-polarity,
                    precursor_polarity=polarity,
                    precursor_lag_state=previous.rate_state,
                    terminal_type="SAME_DATE_ALIGNMENT",
                    terminal_age_dates=0,
                    units=current.rank.units,
                    dominant_collateral_spread_venue=(
                        current.rank.feature.dominant_collateral_spread_venue
                    ),
                )
            )
        previous = current
    return output


def stale_state_rows(rows: Sequence[StateRow], lag: int) -> list[StateRow]:
    if lag <= 0:
        raise RuntimeError("RMSR stale lag must be positive")
    output: list[StateRow] = []
    segment = 0
    previous_index: int | None = None
    previous_source_epoch: int | None = None
    for index, current in enumerate(rows):
        source_index = index - lag
        if source_index < 0 or rows[source_index].epoch != current.epoch:
            previous_index = None
            previous_source_epoch = None
            continue
        stale = rows[source_index]
        contiguous = (
            previous_index is not None
            and previous_index == index - 1
            and previous_source_epoch == current.epoch
        )
        if not contiguous:
            segment += 1
        units = {
            "mix_disagreement": stale.rank.units["mix_disagreement"],
            "rate_disagreement": stale.rank.units["rate_disagreement"],
        }
        rank = RankRow(feature=current.rank.feature, units=units)
        output.append(
            StateRow(
                rank=rank,
                epoch=segment,
                mix_state=component_state(units["mix_disagreement"]),
                rate_state=component_state(units["rate_disagreement"]),
            )
        )
        previous_index = index
        previous_source_epoch = current.epoch
    return output


def permuted_rate_state_rows(rows: Sequence[StateRow]) -> list[StateRow]:
    output: list[StateRow] = []
    by_year: dict[int, list[StateRow]] = defaultdict(list)
    for row in rows:
        by_year[row.rank.feature.observation_date.year].append(row)
    for year in sorted(by_year):
        destinations = sorted(
            by_year[year], key=lambda row: row.rank.feature.observation_date
        )
        sources = sorted(
            destinations,
            key=lambda row: hashlib.sha256(
                (
                    "RMSR-72|year_rate_permutation|"
                    f"{year}|{row.rank.feature.observation_date.isoformat()}"
                ).encode()
            ).digest(),
        )
        for destination, source in zip(destinations, sources):
            units = dict(destination.rank.units)
            units["rate_disagreement"] = source.rank.units["rate_disagreement"]
            output.append(
                StateRow(
                    rank=RankRow(feature=destination.rank.feature, units=units),
                    epoch=destination.epoch,
                    mix_state=destination.mix_state,
                    rate_state=component_state(units["rate_disagreement"]),
                )
            )
    return sorted(output, key=lambda row: row.rank.feature.observation_date)


def _ceil_5m(value: datetime) -> datetime:
    seconds = int(value.timestamp())
    interval = int(BAR.total_seconds())
    return datetime.fromtimestamp(((seconds + interval - 1) // interval) * interval, UTC)


def _split(entry: datetime, exit_time: datetime) -> str | None:
    if TRAIN_START <= entry and exit_time <= TRAIN_END:
        return "train"
    if SELECTION_START <= entry and exit_time <= SELECTION_END:
        return "selection"
    return None


def schedule(control: str, candidates: Sequence[RaceCandidate]) -> list[Scheduled]:
    output: list[Scheduled] = []
    reserved_until: datetime | None = None
    for candidate in sorted(
        candidates,
        key=lambda row: (
            row.signal_time,
            row.terminal_observation_date,
            row.precursor_observation_date,
        ),
    ):
        entry = _ceil_5m(candidate.signal_time) + BAR
        exit_time = entry + HOLD
        split = _split(entry, exit_time)
        if split is None:
            continue
        if reserved_until is not None and entry < reserved_until:
            continue
        output.append(
            Scheduled(
                control=control,
                precursor_observation_date=candidate.precursor_observation_date,
                terminal_observation_date=candidate.terminal_observation_date,
                signal_time=candidate.signal_time,
                entry_time=entry,
                exit_time=exit_time,
                split=split,
                side=candidate.side,
                precursor_polarity=candidate.precursor_polarity,
                precursor_lag_state=candidate.precursor_lag_state,
                terminal_type=candidate.terminal_type,
                terminal_age_dates=candidate.terminal_age_dates,
                units=candidate.units,
                dominant_collateral_spread_venue=(
                    candidate.dominant_collateral_spread_venue
                ),
            )
        )
        reserved_until = exit_time
    return output


def _random_side(entry: datetime) -> int:
    key = f"RMSR-72|deterministic_random_side|{entry.isoformat()}".encode()
    return 1 if hashlib.sha256(key).digest()[0] < 128 else -1


def _clone_clock(
    rows: Sequence[Scheduled], control: str, side: Callable[[Scheduled], int]
) -> list[Scheduled]:
    return [replace(row, control=control, side=side(row)) for row in rows]


def build_clocks(
    state_rows: Sequence[StateRow],
) -> tuple[dict[str, list[Scheduled]], dict[str, dict[str, int]]]:
    primary_candidates, primary_audit = derive_race_candidates(state_rows)
    primary = schedule("primary", primary_candidates)
    reverse_candidates, reverse_audit = derive_race_candidates(
        state_rows, lead="rate", lag="mix"
    )
    five_candidates, five_audit = derive_race_candidates(state_rows, window=5)
    forty_candidates, forty_audit = derive_race_candidates(state_rows, window=40)
    stale_one_candidates, stale_one_audit = derive_race_candidates(
        stale_state_rows(state_rows, 1)
    )
    stale_five_candidates, stale_five_audit = derive_race_candidates(
        stale_state_rows(state_rows, 5)
    )
    permutation_candidates, permutation_audit = derive_race_candidates(
        permuted_rate_state_rows(state_rows)
    )
    clocks: dict[str, list[Scheduled]] = {
        "primary": primary,
        "mix_transition_only": schedule(
            "mix_transition_only",
            _transition_candidates(state_rows, component="mix"),
        ),
        "rate_transition_only": schedule(
            "rate_transition_only",
            _transition_candidates(state_rows, component="rate"),
        ),
        "price_confirmation_only": schedule(
            "price_confirmation_only",
            [
                row
                for row in primary_candidates
                if row.terminal_type == "PRICE_CONFIRMATION"
            ],
        ),
        "quantity_absorption_only": schedule(
            "quantity_absorption_only",
            [
                row
                for row in primary_candidates
                if row.terminal_type == "QUANTITY_ABSORPTION"
            ],
        ),
        "reverse_race": schedule("reverse_race", reverse_candidates),
        "five_date_window": schedule("five_date_window", five_candidates),
        "forty_date_window": schedule("forty_date_window", forty_candidates),
        "one_complete_date_stale": schedule(
            "one_complete_date_stale", stale_one_candidates
        ),
        "five_complete_date_stale": schedule(
            "five_complete_date_stale", stale_five_candidates
        ),
        "year_rate_permutation": schedule(
            "year_rate_permutation", permutation_candidates
        ),
        "same_date_alignment": schedule(
            "same_date_alignment", _same_date_alignment_candidates(state_rows)
        ),
    }
    clocks.update(
        {
            "exact_direction_flip": _clone_clock(
                primary, "exact_direction_flip", lambda row: -row.side
            ),
            "deterministic_random_side": _clone_clock(
                primary,
                "deterministic_random_side",
                lambda row: _random_side(row.entry_time),
            ),
            "constant_long": _clone_clock(
                primary, "constant_long", lambda _row: 1
            ),
            "constant_short": _clone_clock(
                primary, "constant_short", lambda _row: -1
            ),
        }
    )
    audits = {
        "primary": primary_audit,
        "reverse_race": reverse_audit,
        "five_date_window": five_audit,
        "forty_date_window": forty_audit,
        "one_complete_date_stale": stale_one_audit,
        "five_complete_date_stale": stale_five_audit,
        "year_rate_permutation": permutation_audit,
    }
    if set(clocks) != set(CONTROL_NAMES):
        raise RuntimeError("RMSR control clock set changed")
    return clocks, audits


def _contained(
    rows: Sequence[Scheduled], start: datetime, end: datetime
) -> list[Scheduled]:
    return [row for row in rows if row.entry_time >= start and row.exit_time <= end]


def _summary(
    rows: Sequence[Scheduled], start: datetime, end: datetime
) -> dict[str, Any]:
    selected = _contained(rows, start, end)
    month_counts = Counter(row.entry_time.strftime("%Y-%m") for row in selected)
    quarter_counts = Counter(
        f"{row.entry_time.year}-Q{(row.entry_time.month - 1) // 3 + 1}"
        for row in selected
    )
    terminal_counts = Counter(row.terminal_type for row in selected)
    gaps = [
        (current.entry_time - previous.entry_time).total_seconds() / 86_400
        for previous, current in zip(selected, selected[1:])
    ]
    return {
        "events": len(selected),
        "longs": sum(row.side == 1 for row in selected),
        "shorts": sum(row.side == -1 for row in selected),
        "active_months": len(month_counts),
        "active_quarters": len(quarter_counts),
        "max_single_month_share": (
            max(month_counts.values()) / len(selected) if selected else 0.0
        ),
        "maximum_entry_gap_elapsed_days": max(gaps, default=0.0),
        "terminal_type_counts": dict(sorted(terminal_counts.items())),
        "terminal_type_shares": (
            {
                name: count / len(selected)
                for name, count in sorted(terminal_counts.items())
            }
            if selected
            else {}
        ),
    }


def _clock_valid(control: str, rows: Sequence[Scheduled]) -> bool:
    return (
        all(row.control == control for row in rows)
        and all(row.side in {-1, 1} for row in rows)
        and all(row.signal_time < row.entry_time < row.exit_time for row in rows)
        and all(row.entry_time == _ceil_5m(row.signal_time) + BAR for row in rows)
        and all(row.exit_time - row.entry_time == HOLD for row in rows)
        and all(_split(row.entry_time, row.exit_time) == row.split for row in rows)
        and len({row.entry_time for row in rows}) == len(rows)
        and all(
            current.entry_time >= previous.exit_time
            for previous, current in zip(rows, rows[1:])
        )
    )


def _primary_race_valid(rows: Sequence[Scheduled]) -> bool:
    return all(
        row.precursor_observation_date < row.terminal_observation_date
        and 1 <= row.terminal_age_dates <= PRIMARY_WINDOW
        and row.precursor_polarity in {-1, 1}
        and row.precursor_lag_state != row.precursor_polarity
        and row.terminal_type in {"PRICE_CONFIRMATION", "QUANTITY_ABSORPTION"}
        and set(row.units) == set(prereg.COMPONENTS)
        and all(isinstance(value, Fraction) for value in row.units.values())
        for row in rows
    )


def _dominance_distribution(rows: Sequence[Scheduled]) -> dict[str, Any]:
    counts = Counter(row.dominant_collateral_spread_venue for row in rows)
    total = sum(counts.values())
    non_tie = {name: count for name, count in counts.items() if not name.startswith("TIE:")}
    non_tie_total = sum(non_tie.values())
    return {
        "counts": dict(sorted(counts.items())),
        "shares": (
            {name: count / total for name, count in sorted(counts.items())}
            if total
            else {}
        ),
        "non_tie_maximum_share": (
            max(non_tie.values()) / non_tie_total if non_tie_total else 0.0
        ),
    }


def source_support(
    primary: Sequence[Scheduled],
    policy: Mapping[str, Any],
    *,
    post_2023_source_rows: int = 0,
) -> tuple[dict[str, bool], dict[str, dict[str, Any]], dict[str, Any]]:
    summaries = {
        "train_selection": _summary(primary, TRAIN_START, SELECTION_END),
        "train": _summary(primary, TRAIN_START, TRAIN_END),
        "selection": _summary(primary, SELECTION_START, SELECTION_END),
        "2021": _summary(primary, TRAIN_START, datetime(2022, 1, 1, tzinfo=UTC)),
        "2022": _summary(primary, datetime(2022, 1, 1, tzinfo=UTC), TRAIN_END),
        "train_h1_2021": _summary(
            primary, TRAIN_START, datetime(2021, 7, 1, tzinfo=UTC)
        ),
        "train_h2_2021": _summary(
            primary,
            datetime(2021, 7, 1, tzinfo=UTC),
            datetime(2022, 1, 1, tzinfo=UTC),
        ),
        "train_h1_2022": _summary(
            primary,
            datetime(2022, 1, 1, tzinfo=UTC),
            datetime(2022, 7, 1, tzinfo=UTC),
        ),
        "train_h2_2022": _summary(
            primary, datetime(2022, 7, 1, tzinfo=UTC), TRAIN_END
        ),
        "selection_h1": _summary(
            primary, SELECTION_START, datetime(2023, 7, 1, tzinfo=UTC)
        ),
        "selection_h2": _summary(
            primary, datetime(2023, 7, 1, tzinfo=UTC), SELECTION_END
        ),
    }
    dominance = {
        "train": _dominance_distribution(_contained(primary, TRAIN_START, TRAIN_END)),
        "selection": _dominance_distribution(
            _contained(primary, SELECTION_START, SELECTION_END)
        ),
    }
    gates = policy["source_support_gates"]

    def terminal_share(split: str, terminal: str) -> float:
        return summaries[split]["terminal_type_shares"].get(terminal, 0.0)

    checks = {
        "primary_clock_valid": _clock_valid("primary", primary),
        "primary_race_valid": _primary_race_valid(primary),
        "train_total": summaries["train"]["events"] >= gates["train_total_minimum"],
        "each_train_year": all(
            summaries[year]["events"] >= gates["each_train_year_minimum"]
            for year in ("2021", "2022")
        ),
        "each_train_half": all(
            summaries[name]["events"] >= gates["each_train_half_minimum"]
            for name in (
                "train_h1_2021",
                "train_h2_2021",
                "train_h1_2022",
                "train_h2_2022",
            )
        ),
        "train_each_side": min(
            summaries["train"]["longs"], summaries["train"]["shorts"]
        )
        >= gates["train_each_side_minimum"],
        "selection_total": summaries["selection"]["events"]
        >= gates["selection_total_minimum"],
        "each_selection_half": all(
            summaries[name]["events"] >= gates["each_selection_half_minimum"]
            for name in ("selection_h1", "selection_h2")
        ),
        "selection_each_side": min(
            summaries["selection"]["longs"], summaries["selection"]["shorts"]
        )
        >= gates["selection_each_side_minimum"],
        "every_quarter_active": (
            summaries["train"]["active_quarters"] == 8
            and summaries["selection"]["active_quarters"] == 4
        ),
        "train_month_concentration": summaries["train"]["max_single_month_share"]
        <= gates["train_maximum_month_share"],
        "selection_month_concentration": summaries["selection"][
            "max_single_month_share"
        ]
        <= gates["selection_maximum_month_share"],
        "maximum_entry_gap": summaries["train_selection"][
            "maximum_entry_gap_elapsed_days"
        ]
        <= gates["maximum_accepted_entry_gap_elapsed_days"],
        "train_each_terminal_type": min(
            terminal_share("train", "PRICE_CONFIRMATION"),
            terminal_share("train", "QUANTITY_ABSORPTION"),
        )
        >= gates["train_each_terminal_type_minimum_share"],
        "selection_each_terminal_type": min(
            terminal_share("selection", "PRICE_CONFIRMATION"),
            terminal_share("selection", "QUANTITY_ABSORPTION"),
        )
        >= gates["selection_each_terminal_type_minimum_share"],
        "dominant_venue_concentration": all(
            row["non_tie_maximum_share"]
            <= gates["maximum_non_tie_dominant_rate_spread_venue_share"]
            for row in dominance.values()
        ),
        "accepted_ambiguity_count": sum(
            row.terminal_type == "AMBIGUOUS_SAME_DATE" for row in primary
        )
        == gates["accepted_ambiguity_count_required"],
        "post_2023_source_rows": post_2023_source_rows
        == gates["post_2023_source_rows_read_required"],
    }
    return checks, summaries, dominance


def load_comparator_groups() -> tuple[dict[str, list[ComparatorEvent]], int]:
    if tuple(prereg.COMPARATOR_SPECS[:-1]) != tuple(
        rvfc_support.prereg.COMPARATOR_SPECS
    ):
        raise ComparatorValidationError(
            "RMSR inherited comparator cohort drifted from preregistration",
            rows_read=0,
        )
    groups: dict[str, list[ComparatorEvent]] = defaultdict(list)
    rows_read = 0
    for index, spec in enumerate(prereg.COMPARATOR_SPECS):
        name = str(spec["name"])
        try:
            if sha256_file(spec["path"]) != spec["sha256"]:
                raise RuntimeError(f"comparator hash mismatch: {name}")
            included_for_spec = 0
            with gzip.open(_repository_path(spec["path"]), "rt", newline="") as handle:
                reader = csv.DictReader(handle)
                expected_header = (
                    rvfc_support.CLOCK_COLUMNS
                    if index == len(prereg.COMPARATOR_SPECS) - 1
                    else rvfc_support.COMPARATOR_HEADERS[name]
                )
                if tuple(reader.fieldnames or ()) != expected_header:
                    raise RuntimeError(f"comparator header changed: {name}")
                for raw in reader:
                    rows_read += 1
                    if index == len(prereg.COMPARATOR_SPECS) - 1:
                        group = "rvfc:primary"
                        entry_field = "entry_time"
                        exit_field = "exit_time"
                        include = raw["control"] == "primary"
                    else:
                        group, entry_field, exit_field, include = (
                            rvfc_support._comparator_identity(name, raw)
                        )
                    if not include:
                        continue
                    included_for_spec += 1
                    event = ComparatorEvent(
                        entry_time=_timestamp(raw[entry_field]),
                        exit_time=_timestamp(raw[exit_field]),
                        side=_side(raw["side"]),
                    )
                    if event.exit_time <= event.entry_time:
                        raise RuntimeError(f"invalid comparator interval: {name}")
                    if (
                        event.entry_time >= SELECTION_END
                        or event.exit_time > SELECTION_END
                    ):
                        raise RuntimeError(f"post-2023 comparator clock: {name}")
                    groups[group].append(event)
            if included_for_spec == 0:
                raise RuntimeError(f"required comparator is empty: {name}")
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            raise ComparatorValidationError(str(exc), rows_read=rows_read) from exc
    if not groups:
        raise ComparatorValidationError(
            "RMSR comparator cohort is empty", rows_read=rows_read
        )
    normalized: dict[str, list[ComparatorEvent]] = {}
    for name, events in sorted(groups.items()):
        rows = sorted(events, key=lambda row: row.entry_time)
        if len({row.entry_time for row in rows}) != len(rows):
            raise ComparatorValidationError(
                f"duplicate comparator entry: {name}", rows_read=rows_read
            )
        if any(
            current.entry_time < previous.exit_time
            for previous, current in zip(rows, rows[1:])
        ):
            raise ComparatorValidationError(
                f"overlapping comparator: {name}", rows_read=rows_read
            )
        normalized[name] = rows
    return normalized, rows_read


def one_to_one_matches(
    left: Sequence[datetime], right: Sequence[datetime], tolerance: timedelta
) -> int:
    left = sorted(left)
    right = sorted(right)
    i = j = matches = 0
    while i < len(left) and j < len(right):
        delta = left[i] - right[j]
        if abs(delta) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif delta < timedelta(0):
            i += 1
        else:
            j += 1
    return matches


def _exposure(
    events: Sequence[ComparatorEvent], start: datetime, end: datetime
) -> np.ndarray:
    size = int((end - start).total_seconds() // BAR.total_seconds())
    values = np.zeros(size, dtype=np.float64)
    for event in events:
        left = max(event.entry_time, start)
        right = min(event.exit_time, end)
        if right <= left:
            continue
        begin = int((left - start).total_seconds() // BAR.total_seconds())
        finish = int((right - start).total_seconds() // BAR.total_seconds())
        values[begin:finish] += event.side
    return values


def novelty_metrics(
    primary: Sequence[Scheduled], comparator: Sequence[ComparatorEvent]
) -> dict[str, Any]:
    primary_events = [
        ComparatorEvent(row.entry_time, row.exit_time, row.side)
        for row in primary
        if row.entry_time >= TRAIN_START and row.exit_time <= SELECTION_END
    ]
    comparator_events = [
        row
        for row in comparator
        if row.entry_time >= TRAIN_START and row.exit_time <= SELECTION_END
    ]
    a = [row.entry_time for row in primary_events]
    b = [row.entry_time for row in comparator_events]
    exact_a, exact_b = set(a), set(b)
    union = exact_a | exact_b
    matches = one_to_one_matches(a, b, timedelta(hours=24))
    left_exposure = _exposure(primary_events, TRAIN_START, SELECTION_END)
    right_exposure = _exposure(comparator_events, TRAIN_START, SELECTION_END)
    correlation: float | None
    if np.std(left_exposure) == 0 or np.std(right_exposure) == 0:
        correlation = None
    else:
        correlation = float(np.corrcoef(left_exposure, right_exposure)[0, 1])
        if not math.isfinite(correlation):
            correlation = None
    return {
        "primary_entries": len(a),
        "comparator_entries": len(b),
        "exact_entry_intersection": len(exact_a & exact_b),
        "exact_entry_jaccard": len(exact_a & exact_b) / len(union) if union else 0.0,
        "one_day_one_to_one_matches": matches,
        "rmsr_one_day_containment": matches / len(a) if a else 0.0,
        "comparator_one_day_containment": matches / len(b) if b else 0.0,
        "signed_5m_occupied_exposure_correlation": correlation,
    }


def _clock_row(row: Scheduled) -> dict[str, str]:
    return {
        "control": row.control,
        "precursor_observation_date": row.precursor_observation_date.isoformat(),
        "terminal_observation_date": row.terminal_observation_date.isoformat(),
        "signal_time": row.signal_time.isoformat(),
        "entry_time": row.entry_time.isoformat(),
        "exit_time": row.exit_time.isoformat(),
        "split": row.split,
        "side": str(row.side),
        "precursor_polarity": str(row.precursor_polarity),
        "precursor_lag_state": str(row.precursor_lag_state),
        "terminal_type": row.terminal_type,
        "terminal_age_dates": str(row.terminal_age_dates),
        "u_mix_disagreement": str(row.units["mix_disagreement"]),
        "u_rate_disagreement": str(row.units["rate_disagreement"]),
        "dominant_collateral_spread_venue": (
            row.dominant_collateral_spread_venue
        ),
    }


def _gzip_csv(rows: Iterable[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(CLOCK_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return gzip.compress(buffer.getvalue().encode(), compresslevel=9, mtime=0)


def _atomic_write(path: Path, payload: bytes) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RMSR output escaped repository") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_or_verify(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing RMSR artifact differs: {path.name}")
        return "verified_existing"
    try:
        _atomic_write(path, payload)
        return "created"
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RuntimeError(f"concurrent RMSR artifact differs: {path.name}")
        return "verified_existing"


def build_report(
    *,
    clock_output: str | Path = DEFAULT_CLOCK,
    load_comparators: Callable[
        [], tuple[dict[str, list[ComparatorEvent]], int]
    ] = load_comparator_groups,
    write_clock: bool = True,
) -> dict[str, Any]:
    registration = _load_registration()
    by_date, normalized_rows, required_rows = load_source()
    features, source_audit = build_features(
        by_date,
        normalized_rows_read=normalized_rows,
        required_rows_read=required_rows,
    )
    rank_rows = build_rank_rows(features)
    state_rows = build_state_rows(rank_rows)
    clocks, race_audits = build_clocks(state_rows)
    if not all(_clock_valid(name, rows) for name, rows in clocks.items()):
        raise RuntimeError("RMSR generated an invalid control clock")
    post_2023_source_rows = sum(day > END_DATE for day in by_date)
    source_checks, summaries, dominance = source_support(
        clocks["primary"],
        registration["policy"],
        post_2023_source_rows=post_2023_source_rows,
    )
    source_passed = all(source_checks.values())
    comparator_rows_read = 0
    novelty: dict[str, Any] = {
        "evaluated": False,
        "reason": "source support failed before comparator access",
        "metrics": {},
        "checks": {},
        "qualifying_groups": [],
        "passed": False,
    }
    if source_passed:
        try:
            groups, comparator_rows_read = load_comparators()
        except ComparatorValidationError as exc:
            comparator_rows_read = exc.rows_read
            novelty = {
                "evaluated": False,
                "reason": "comparator validation failed closed",
                "validation_error": str(exc),
                "metrics": {},
                "checks": {},
                "qualifying_groups": [],
                "passed": False,
            }
        else:
            metrics = {
                name: novelty_metrics(clocks["primary"], rows)
                for name, rows in groups.items()
            }
            limits = registration["policy"]["novelty"]
            minimum = limits["minimum_comparator_entries"]
            qualifying = sorted(
                name
                for name, row in metrics.items()
                if row["comparator_entries"] >= minimum
            )
            if not qualifying:
                novelty = {
                    "evaluated": False,
                    "reason": "no qualifying comparator groups",
                    "metrics": metrics,
                    "checks": {},
                    "qualifying_groups": [],
                    "passed": False,
                }
            else:
                checks: dict[str, dict[str, bool]] = {}
                for name in qualifying:
                    row = metrics[name]
                    correlation = row["signed_5m_occupied_exposure_correlation"]
                    checks[name] = {
                        "exact_entry_jaccard": row["exact_entry_jaccard"]
                        <= limits["maximum_exact_entry_jaccard"],
                        "one_day_containment": row["rmsr_one_day_containment"]
                        <= limits["maximum_rmsr_one_day_containment"],
                        "signed_exposure_correlation": correlation is not None
                        and abs(correlation)
                        <= limits["maximum_absolute_signed_exposure_correlation"],
                    }
                novelty = {
                    "evaluated": True,
                    "reason": "source support passed",
                    "metrics": metrics,
                    "checks": checks,
                    "qualifying_groups": qualifying,
                    "passed": all(all(row.values()) for row in checks.values()),
                }
    clock_payload = _gzip_csv(
        _clock_row(row) for control in CONTROL_NAMES for row in clocks[control]
    )
    if write_clock:
        _write_or_verify(_repository_path(clock_output), clock_payload)
    control_summaries = {
        name: {
            "train": _summary(rows, TRAIN_START, TRAIN_END),
            "selection": _summary(rows, SELECTION_START, SELECTION_END),
        }
        for name, rows in clocks.items()
    }
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": prereg.POLICY_ID,
        "support_builder": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "source_audit": asdict(source_audit),
        "rank_ready_rows": len(rank_rows),
        "decision_state_rows": len(state_rows),
        "race_audits": race_audits,
        "clock_summaries": control_summaries,
        "primary_support_summaries": summaries,
        "source_checks": source_checks,
        "source_support_passed": source_passed,
        "dominance_diagnostics": dominance,
        "novelty": novelty,
        "clock_artifact": {
            "path": str(clock_output),
            "sha256": hashlib.sha256(clock_payload).hexdigest(),
            "rows": sum(len(rows) for rows in clocks.values()),
            "primary_rows": len(clocks["primary"]),
            "columns": list(CLOCK_COLUMNS),
        },
        "outcome_boundary": {
            "source_observation_rows_read": normalized_rows,
            "exact_race_incidence_opened": True,
            "comparator_rows_read": comparator_rows_read,
            "comparator_access_short_circuited_on_source_failure": not source_passed,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
            "post_2023_source_rows_read": post_2023_source_rows,
        },
        "advance_to_evaluator_freeze": bool(source_passed and novelty["passed"]),
        "disposition": (
            "advance RMSR-72 unchanged to evaluator freeze"
            if source_passed and novelty["passed"]
            else "reject RMSR-72 unchanged before outcomes"
        ),
    }
    core["manifest_hash"] = canonical_hash(core)
    return core


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    args = parser.parse_args(argv)
    report = build_report(clock_output=args.clock_output)
    _write_or_verify(
        _repository_path(args.report_output),
        (
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode(),
    )
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "source_support_passed": report["source_support_passed"],
                "novelty_evaluated": report["novelty"]["evaluated"],
                "novelty_passed": report["novelty"]["passed"],
                "advance_to_evaluator_freeze": report[
                    "advance_to_evaluator_freeze"
                ],
                "manifest_hash": report["manifest_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
