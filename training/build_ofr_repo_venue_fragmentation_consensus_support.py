"""Build outcome-blind RVFC-72 source-support, control, and novelty clocks."""

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

from training import preregister_ofr_repo_venue_fragmentation_consensus as prereg


PROTOCOL_VERSION = "ofr_repo_venue_fragmentation_consensus_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_ofr_repo_venue_fragmentation_consensus_support.py")
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "681c9c15a897f277ff45c45ffe4144cda018b17ad2f698e06494c03f82948380"
)
PREREGISTRATION_MANIFEST_HASH = (
    "58c1cc02489ed7844fefb0eab5a9831f8104dbd57172b3faa96bc69fddb71104"
)
DEFAULT_CLOCK = Path(
    "results/ofr_repo_venue_fragmentation_consensus_clocks_2026-07-23.csv.gz"
)
DEFAULT_REPORT = Path(
    "results/ofr_repo_venue_fragmentation_consensus_support_2026-07-23.json"
)

UTC = timezone.utc
BAR = timedelta(minutes=5)
HOLD = timedelta(hours=72)
LOOKBACK = 252
START_DATE = date(2019, 1, 1)
END_DATE = date(2023, 12, 31)
FEED_FLOOR = datetime(2020, 9, 10, tzinfo=UTC)
TRAIN_START = datetime(2021, 1, 1, tzinfo=UTC)
TRAIN_END = datetime(2023, 1, 1, tzinfo=UTC)
SELECTION_START = TRAIN_END
SELECTION_END = datetime(2024, 1, 1, tzinfo=UTC)

COMPONENT_ONLY = tuple(f"{name}_only" for name in prereg.COMPONENTS)
LEAVE_ONE = tuple(f"leave_one_{name}" for name in prereg.COMPONENTS)
SOURCE_CONTROLS = (
    *COMPONENT_ONLY,
    "mean_without_consensus",
    "same_sign_without_magnitude",
    "rate_family_only",
    "volume_family_only",
    *LEAVE_ONE,
    "one_complete_day_stale",
    "five_complete_day_stale",
    "year_component_permutation",
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
    "observation_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "split",
    "side",
    "state",
    "score_fraction",
    "u_rate_dispersion",
    "u_venue_hhi",
    "u_collateral_rate_disagreement",
    "u_collateral_mix_disagreement",
    "dominant_rate_venue",
    "dominant_volume_venue",
    "dominant_collateral_spread_venue",
)

COMPARATOR_HEADERS: Mapping[str, tuple[str, ...]] = {
    "overnight_rrp_flow_release_all_controls": (
        "control", "signal_time", "entry_time", "exit_time", "side", "operation_date"
    ),
    "overnight_rrp_participant_breadth_all_controls": (
        "candidate_id", "control", "split", "origin_operation_date",
        "operation_date", "decision_time", "entry_time", "exit_time", "side",
        "score", "rank",
    ),
    "federal_liquidity_component_concordance_all_groups": (
        "candidate_id", "clock_name", "feature_release_date", "signal_release_date",
        "signal_time", "entry_time", "exit_time", "side", "horizon_releases",
        "lower_rank_numerator", "upper_rank_numerator", "prior_lookback",
        "net_rank_numerator", "asset_rank_numerator", "tga_release_rank_numerator",
        "rrp_release_rank_numerator", "component_breadth", "component_tail_breadth",
    ),
    "daily_treasury_fiscal_flow_breadth_primary": (
        "policy_id", "clock", "window", "signal_record_date", "execution_record_date",
        "decision_time_utc", "entry_time_utc", "exit_time_utc", "side",
        "deposit_breadth", "withdrawal_breadth", "issue_breadth", "redemption_breadth",
        "deposit_eligible_categories", "withdrawal_eligible_categories",
        "issue_eligible_categories", "redemption_eligible_categories", "cash_impulse",
        "debt_impulse", "cash_rank126", "debt_rank126", "total_net_cash",
        "total_net_cash_rank126",
    ),
    "daily_treasury_fiscal_flow_breadth_controls": (
        "policy_id", "clock", "window", "signal_record_date", "execution_record_date",
        "decision_time_utc", "entry_time_utc", "exit_time_utc", "side",
        "deposit_breadth", "withdrawal_breadth", "issue_breadth", "redemption_breadth",
        "deposit_eligible_categories", "withdrawal_eligible_categories",
        "issue_eligible_categories", "redemption_eligible_categories", "cash_impulse",
        "debt_impulse", "cash_rank126", "debt_rank126", "total_net_cash",
        "total_net_cash_rank126",
    ),
    "sofr_rate_dislocation_primary": (
        "event_index", "effective_date", "sofr_available_at_utc", "delta_bp",
        "rank_twice_numerator", "rank_twice_denominator", "state", "side",
        "entry_time", "exit_time",
    ),
    "bank_deposit_secured_repo_concordance_all_clocks": (
        "clock_name", "release_date", "decision_time", "entry_time", "exit_time",
        "side", "h8_sign", "repo_sign", "repo5_bp", "sofr_effective_date",
        "sofr_available_at",
    ),
    "fed_h8_deposit_migration_primary": (
        "release_date", "signal_time", "entry_time", "exit_time", "side",
        "clock_mode", "adjustment", "tail_quantile", "migration_bp",
        "small_borrowings_bp", "small_cash_bp", "migration_z", "borrowings_z",
        "cash_stress_z", "stress_score", "agreement_count", "threshold_abs",
    ),
    "soma_lending_collateral_scarcity_primary": (
        "control", "operation_id", "operation_date", "signal_time", "entry_time",
        "exit_time", "split", "side", "state", "score", "u_demand_intensity",
        "u_weighted_fee", "u_carry_intensity", "u_demand_breadth",
    ),
    "cross_domain_liquidity_transmission_all_clocks": (
        "clock", "window", "decision_time_utc", "entry_time_utc", "exit_time_utc",
        "side",
    ),
    "live_portfolio_pure_clocks": (
        "candidate_id", "split", "decision_time", "entry_time", "exit_time", "side"
    ),
}


@dataclass(frozen=True)
class SourceRow:
    mnemonic: str
    observation_date: date
    available_at: datetime
    value: Fraction | None


@dataclass(frozen=True)
class FeatureRow:
    observation_date: date
    available_at: datetime
    epoch: int
    decision_allowed: bool
    components: Mapping[str, Fraction]
    dominant_rate_venue: str
    dominant_volume_venue: str
    dominant_collateral_spread_venue: str


@dataclass(frozen=True)
class RankRow:
    feature: FeatureRow
    units: Mapping[str, Fraction]


@dataclass(frozen=True)
class StatePoint:
    control: str
    rank: RankRow
    state: int
    score: Fraction


@dataclass(frozen=True)
class Candidate:
    control: str
    rank: RankRow
    state: int
    side: int
    score: Fraction


@dataclass(frozen=True)
class Scheduled:
    control: str
    observation_date: date
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    split: str
    side: int
    state: int
    score: Fraction
    units: Mapping[str, Fraction]
    dominant_rate_venue: str
    dominant_volume_venue: str
    dominant_collateral_spread_venue: str


@dataclass(frozen=True)
class ComparatorEvent:
    entry_time: datetime
    exit_time: datetime
    side: int


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
        raise RuntimeError("RVFC support path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RVFC support path must remain repository-relative") from exc
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("RVFC timestamp must be text")
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RuntimeError(f"invalid RVFC timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError("RVFC timestamp must be timezone-aware")
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
    raise RuntimeError(f"invalid RVFC comparator side: {value}")


def _expected_availability(day: date) -> datetime:
    delayed = datetime.combine(day + timedelta(days=8), time(), UTC)
    return max(delayed, FEED_FLOOR)


def _load_registration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("RVFC preregistration file hash mismatch")
    payload = json.loads(_repository_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_preregistration(payload, verify_sources=True)
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("RVFC preregistration manifest hash mismatch")
    return payload


def load_source() -> tuple[dict[date, dict[str, SourceRow]], int, int]:
    if sha256_file(prereg.OBSERVATIONS) != prereg.OBSERVATIONS_SHA256:
        raise RuntimeError("RVFC source observation hash mismatch")
    required = set(prereg.REQUIRED_SERIES)
    by_date: dict[date, dict[str, SourceRow]] = defaultdict(dict)
    all_dates: set[date] = set()
    normalized_rows = required_rows = 0
    with gzip.open(_repository_path(prereg.OBSERVATIONS), "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise RuntimeError("RVFC source columns changed")
        for raw in reader:
            normalized_rows += 1
            try:
                observation_day = date.fromisoformat(raw["observation_date"])
            except ValueError as exc:
                raise RuntimeError("RVFC observation date changed") from exc
            if not START_DATE <= observation_day <= END_DATE:
                raise RuntimeError("RVFC source date escaped frozen window")
            all_dates.add(observation_day)
            mnemonic = raw["mnemonic"]
            if mnemonic not in required:
                continue
            required_rows += 1
            available_at = _timestamp(raw["available_at_utc"])
            if available_at != _expected_availability(observation_day):
                raise RuntimeError("RVFC source availability changed")
            if mnemonic in by_date[observation_day]:
                raise RuntimeError("duplicate RVFC required source row")
            value = _fraction(raw["value"], f"{mnemonic} value", optional=True)
            by_date[observation_day][mnemonic] = SourceRow(
                mnemonic=mnemonic,
                observation_date=observation_day,
                available_at=available_at,
                value=value,
            )
    if normalized_rows != 77_369:
        raise RuntimeError("RVFC normalized source row count changed")
    for observation_day in all_dates:
        by_date.setdefault(observation_day, {})
    if not by_date:
        raise RuntimeError("RVFC required source is empty")
    return dict(by_date), normalized_rows, required_rows


def _required_value(rows: Mapping[str, SourceRow], mnemonic: str) -> Fraction:
    row = rows[mnemonic]
    if row.value is None:
        raise RuntimeError("required RVFC value is null")
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
        if set(rows) != required or any(row.value is None for row in rows.values()):
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
        venue_volumes = {
            "DVP": value("REPO-DVP_TV_TOT-P"),
            "GCF": value("REPO-GCF_TV_TOT-P"),
            "TRIV1": value("REPO-TRIV1_TV_TOT-P"),
        }
        venue_total = sum(venue_volumes.values(), Fraction())
        material = (
            venue_total > 0
            and gcf_total > 0
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
        venue_rates = {
            "DVP": value("REPO-DVP_AR_TOT-P"),
            "GCF": value("REPO-GCF_AR_TOT-P"),
            "TRIV1": value("REPO-TRIV1_AR_TOT-P"),
        }
        collateral_spreads = {
            "GCF": abs(value("REPO-GCF_AR_AG-P") - value("REPO-GCF_AR_T-P")),
            "TRIV1": abs(
                value("REPO-TRIV1_AR_AG-P") - value("REPO-TRIV1_AR_T-P")
            ),
        }
        components = {
            "rate_dispersion": max(venue_rates.values()) - min(venue_rates.values()),
            "venue_hhi": sum(
                (amount / venue_total) ** 2 for amount in venue_volumes.values()
            ),
            "collateral_rate_disagreement": sum(collateral_spreads.values()) / 2,
            "collateral_mix_disagreement": abs(gcf_ag / gcf_total - tri_ag / tri_total),
        }
        availability = max(row.available_at for row in rows.values())
        features.append(
            FeatureRow(
                observation_date=observation_day,
                available_at=availability,
                epoch=epoch,
                decision_allowed=False,
                components=components,
                dominant_rate_venue=_dominant(venue_rates),
                dominant_volume_venue=_dominant(venue_volumes),
                dominant_collateral_spread_venue=_dominant(collateral_spreads),
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
    suppressed = sum(not row.decision_allowed for row in features)
    audit = SourceAudit(
        normalized_rows_read=normalized_rows_read,
        required_rows_read=required_rows_read,
        source_dates_seen=len(by_date),
        valid_feature_dates=len(features),
        invalid_missing_or_null_dates=invalid_missing_or_null,
        invalid_materiality_dates=invalid_materiality,
        equal_availability_rows_suppressed=suppressed,
    )
    return features, audit


def midrank_unit(current: Fraction, prior: Sequence[Fraction]) -> Fraction:
    if len(prior) != LOOKBACK:
        raise RuntimeError("RVFC midrank history length changed")
    less = sum(value < current for value in prior)
    equal = sum(value == current for value in prior)
    return Fraction(2 * less + equal - LOOKBACK, LOOKBACK)


def build_rank_rows(features: Sequence[FeatureRow]) -> list[RankRow]:
    histories = {name: deque(maxlen=LOOKBACK) for name in prereg.COMPONENTS}
    output: list[RankRow] = []
    for row in features:
        if all(len(history) == LOOKBACK for history in histories.values()):
            output.append(
                RankRow(
                    feature=row,
                    units={
                        name: midrank_unit(row.components[name], tuple(histories[name]))
                        for name in prereg.COMPONENTS
                    },
                )
            )
        for name in prereg.COMPONENTS:
            histories[name].append(row.components[name])
    return output


def _primary_state(units: Mapping[str, Fraction]) -> tuple[int, Fraction]:
    values = tuple(units[name] for name in prereg.COMPONENTS)
    score = sum(values, Fraction()) / len(values)
    if sum(value > 0 for value in values) >= 3 and score >= Fraction(1, 2):
        return 1, score
    if sum(value < 0 for value in values) >= 3 and score <= Fraction(-1, 2):
        return -1, score
    return 0, score


def _state_for_control(
    control: str, units: Mapping[str, Fraction]
) -> tuple[int, Fraction]:
    if control == "primary":
        return _primary_state(units)
    if control.endswith("_only") and control[: -len("_only")] in prereg.COMPONENTS:
        value = units[control[: -len("_only")]]
        return (1 if value >= Fraction(1, 2) else -1 if value <= Fraction(-1, 2) else 0), value
    values = tuple(units[name] for name in prereg.COMPONENTS)
    score = sum(values, Fraction()) / len(values)
    if control == "mean_without_consensus":
        return (1 if score >= Fraction(1, 2) else -1 if score <= Fraction(-1, 2) else 0), score
    if control == "same_sign_without_magnitude":
        if sum(value > 0 for value in values) >= 3:
            return 1, score
        if sum(value < 0 for value in values) >= 3:
            return -1, score
        return 0, score
    if control in {"rate_family_only", "volume_family_only"}:
        names = (
            ("rate_dispersion", "collateral_rate_disagreement")
            if control == "rate_family_only"
            else ("venue_hhi", "collateral_mix_disagreement")
        )
        selected = tuple(units[name] for name in names)
        family_score = sum(selected, Fraction()) / 2
        if all(value >= Fraction(1, 2) for value in selected):
            return 1, family_score
        if all(value <= Fraction(-1, 2) for value in selected):
            return -1, family_score
        return 0, family_score
    if control.startswith("leave_one_"):
        omitted = control[len("leave_one_") :]
        if omitted not in prereg.COMPONENTS:
            raise RuntimeError("unknown RVFC leave-one control")
        selected = tuple(units[name] for name in prereg.COMPONENTS if name != omitted)
        leave_score = sum(selected, Fraction()) / 3
        if sum(value > 0 for value in selected) >= 2 and leave_score >= Fraction(1, 2):
            return 1, leave_score
        if sum(value < 0 for value in selected) >= 2 and leave_score <= Fraction(-1, 2):
            return -1, leave_score
        return 0, leave_score
    raise RuntimeError(f"unknown RVFC source control: {control}")


def _permuted_vectors(rows: Sequence[RankRow]) -> dict[date, dict[str, Fraction]]:
    output = {row.feature.observation_date: {} for row in rows}
    by_year: dict[int, list[RankRow]] = defaultdict(list)
    for row in rows:
        by_year[row.feature.observation_date.year].append(row)
    for year, year_rows in by_year.items():
        destinations = sorted(year_rows, key=lambda row: row.feature.observation_date)
        for component in prereg.COMPONENTS:
            sources = sorted(
                year_rows,
                key=lambda row: hashlib.sha256(
                    (
                        "RVFC-72|year_component_permutation|"
                        f"{year}|{component}|{row.feature.observation_date.isoformat()}"
                    ).encode()
                ).digest(),
            )
            for destination, source in zip(destinations, sources):
                output[destination.feature.observation_date][component] = source.units[
                    component
                ]
    return output


def build_state_points(rank_rows: Sequence[RankRow]) -> dict[str, list[StatePoint]]:
    states: dict[str, list[StatePoint]] = {name: [] for name in ("primary", *SOURCE_CONTROLS)}
    permuted = _permuted_vectors(rank_rows)
    for index, row in enumerate(rank_rows):
        for control in (
            "primary",
            *COMPONENT_ONLY,
            "mean_without_consensus",
            "same_sign_without_magnitude",
            "rate_family_only",
            "volume_family_only",
            *LEAVE_ONE,
        ):
            state, score = _state_for_control(control, row.units)
            states[control].append(StatePoint(control, row, state, score))
        for control, lag in (
            ("one_complete_day_stale", 1),
            ("five_complete_day_stale", 5),
        ):
            if index < lag:
                continue
            stale_units = rank_rows[index - lag].units
            state, score = _primary_state(stale_units)
            states[control].append(
                StatePoint(
                    control,
                    RankRow(row.feature, stale_units),
                    state,
                    score,
                )
            )
        units = permuted[row.feature.observation_date]
        state, score = _primary_state(units)
        states["year_component_permutation"].append(
            StatePoint(
                "year_component_permutation",
                RankRow(row.feature, units),
                state,
                score,
            )
        )
    return states


def candidates_from_states(points: Sequence[StatePoint]) -> list[Candidate]:
    output: list[Candidate] = []
    previous: StatePoint | None = None
    for point in points:
        if previous is None or previous.rank.feature.epoch != point.rank.feature.epoch:
            previous = point
            continue
        if (
            point.rank.feature.decision_allowed
            and point.state in (-1, 1)
            and point.state != previous.state
        ):
            output.append(
                Candidate(
                    control=point.control,
                    rank=point.rank,
                    state=point.state,
                    side=-point.state,
                    score=point.score,
                )
            )
        previous = point
    return output


def _ceil_5m(value: datetime) -> datetime:
    seconds = int(value.timestamp())
    step = int(BAR.total_seconds())
    return datetime.fromtimestamp(((seconds + step - 1) // step) * step, UTC)


def _split(entry: datetime, exit_time: datetime) -> str | None:
    if entry >= TRAIN_START and exit_time <= TRAIN_END:
        return "train"
    if entry >= SELECTION_START and exit_time <= SELECTION_END:
        return "selection"
    return None


def schedule(control: str, candidates: Sequence[Candidate]) -> list[Scheduled]:
    output: list[Scheduled] = []
    last_exit: datetime | None = None
    for candidate in sorted(candidates, key=lambda row: row.rank.feature.available_at):
        signal = candidate.rank.feature.available_at
        entry = _ceil_5m(signal) + BAR
        exit_time = entry + HOLD
        split = _split(entry, exit_time)
        if split is None or (last_exit is not None and entry < last_exit):
            continue
        feature = candidate.rank.feature
        output.append(
            Scheduled(
                control=control,
                observation_date=feature.observation_date,
                signal_time=signal,
                entry_time=entry,
                exit_time=exit_time,
                split=split,
                side=candidate.side,
                state=candidate.state,
                score=candidate.score,
                units=candidate.rank.units,
                dominant_rate_venue=feature.dominant_rate_venue,
                dominant_volume_venue=feature.dominant_volume_venue,
                dominant_collateral_spread_venue=(
                    feature.dominant_collateral_spread_venue
                ),
            )
        )
        last_exit = exit_time
    return output


def _random_side(entry: datetime) -> int:
    digest = hashlib.sha256(
        f"RVFC-72|deterministic_random_side|{entry.isoformat()}".encode()
    ).digest()
    return 1 if digest[0] < 128 else -1


def build_clocks(rank_rows: Sequence[RankRow]) -> dict[str, list[Scheduled]]:
    points = build_state_points(rank_rows)
    clocks = {
        control: schedule(control, candidates_from_states(rows))
        for control, rows in points.items()
    }
    primary = clocks["primary"]
    clocks["exact_direction_flip"] = [
        replace(row, control="exact_direction_flip", side=-row.side) for row in primary
    ]
    clocks["deterministic_random_side"] = [
        replace(
            row,
            control="deterministic_random_side",
            side=_random_side(row.entry_time),
        )
        for row in primary
    ]
    clocks["constant_long"] = [
        replace(row, control="constant_long", side=1) for row in primary
    ]
    clocks["constant_short"] = [
        replace(row, control="constant_short", side=-1) for row in primary
    ]
    if set(clocks) != set(CONTROL_NAMES):
        raise RuntimeError("RVFC control clock set changed")
    return clocks


def _contained(
    rows: Iterable[Scheduled], start: datetime, end: datetime
) -> list[Scheduled]:
    return [row for row in rows if row.entry_time >= start and row.exit_time <= end]


def _summary(rows: Sequence[Scheduled], start: datetime, end: datetime) -> dict[str, Any]:
    selected = _contained(rows, start, end)
    months = Counter(row.entry_time.strftime("%Y-%m") for row in selected)
    quarters = Counter(
        f"{row.entry_time.year}-Q{(row.entry_time.month - 1) // 3 + 1}"
        for row in selected
    )
    gaps = [
        (current.entry_time - previous.entry_time).total_seconds() / 86400
        for previous, current in zip(selected, selected[1:])
    ]
    return {
        "events": len(selected),
        "longs": sum(row.side == 1 for row in selected),
        "shorts": sum(row.side == -1 for row in selected),
        "active_months": len(months),
        "active_quarters": len(quarters),
        "max_single_month_share": (
            max(months.values(), default=0) / len(selected) if selected else 0.0
        ),
        "maximum_entry_gap_elapsed_days": max(gaps, default=0.0),
    }


def _clock_valid(control: str, rows: Sequence[Scheduled]) -> bool:
    dates = [row.observation_date for row in rows]
    entries = [row.entry_time for row in rows]
    return (
        len(dates) == len(set(dates))
        and len(entries) == len(set(entries))
        and all(
            row.control == control
            and row.side in (-1, 1)
            and row.entry_time == _ceil_5m(row.signal_time) + BAR
            and row.exit_time - row.entry_time == HOLD
            and _split(row.entry_time, row.exit_time) == row.split
            for row in rows
        )
        and all(
            current.entry_time >= previous.exit_time
            for previous, current in zip(rows, rows[1:])
        )
    )


def source_support(
    primary: Sequence[Scheduled], policy: Mapping[str, Any]
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    summaries = {
        "train_selection": _summary(primary, TRAIN_START, SELECTION_END),
        "train": _summary(primary, TRAIN_START, TRAIN_END),
        "selection": _summary(primary, SELECTION_START, SELECTION_END),
        "2021": _summary(primary, TRAIN_START, datetime(2022, 1, 1, tzinfo=UTC)),
        "2022": _summary(primary, datetime(2022, 1, 1, tzinfo=UTC), TRAIN_END),
        "train_h1_2021": _summary(primary, TRAIN_START, datetime(2021, 7, 1, tzinfo=UTC)),
        "train_h2_2021": _summary(primary, datetime(2021, 7, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC)),
        "train_h1_2022": _summary(primary, datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 7, 1, tzinfo=UTC)),
        "train_h2_2022": _summary(primary, datetime(2022, 7, 1, tzinfo=UTC), TRAIN_END),
        "selection_h1": _summary(primary, SELECTION_START, datetime(2023, 7, 1, tzinfo=UTC)),
        "selection_h2": _summary(primary, datetime(2023, 7, 1, tzinfo=UTC), SELECTION_END),
    }
    gates = policy["source_support_gates"]
    checks = {
        "primary_clock_valid": _clock_valid("primary", primary),
        "train_total": summaries["train"]["events"] >= gates["train_total_minimum"],
        "each_train_year": all(
            summaries[year]["events"] >= gates["each_train_year_minimum"]
            for year in ("2021", "2022")
        ),
        "each_train_half": all(
            summaries[name]["events"] >= gates["each_train_half_minimum"]
            for name in (
                "train_h1_2021", "train_h2_2021", "train_h1_2022", "train_h2_2022"
            )
        ),
        "train_each_side": min(
            summaries["train"]["longs"], summaries["train"]["shorts"]
        ) >= gates["train_each_side_minimum"],
        "selection_total": summaries["selection"]["events"] >= gates["selection_total_minimum"],
        "each_selection_half": all(
            summaries[name]["events"] >= gates["each_selection_half_minimum"]
            for name in ("selection_h1", "selection_h2")
        ),
        "selection_each_side": min(
            summaries["selection"]["longs"], summaries["selection"]["shorts"]
        ) >= gates["selection_each_side_minimum"],
        "every_quarter_active": (
            summaries["train"]["active_quarters"] == 8
            and summaries["selection"]["active_quarters"] == 4
        ),
        "train_month_concentration": summaries["train"]["max_single_month_share"]
        <= gates["train_maximum_month_share"],
        "selection_month_concentration": summaries["selection"]["max_single_month_share"]
        <= gates["selection_maximum_month_share"],
        "maximum_entry_gap": summaries["train_selection"][
            "maximum_entry_gap_elapsed_days"
        ]
        <= gates["maximum_accepted_entry_gap_elapsed_days"],
    }
    return checks, summaries


def _comparator_identity(name: str, raw: Mapping[str, str]) -> tuple[str, str, str, bool]:
    if name == "overnight_rrp_flow_release_all_controls":
        return f"orfr:{raw['control']}", "entry_time", "exit_time", True
    if name == "overnight_rrp_participant_breadth_all_controls":
        return f"orpb:{raw['control']}", "entry_time", "exit_time", True
    if name == "federal_liquidity_component_concordance_all_groups":
        return f"flcc:{raw['candidate_id']}:{raw['clock_name']}", "entry_time", "exit_time", True
    if name in {
        "daily_treasury_fiscal_flow_breadth_primary",
        "daily_treasury_fiscal_flow_breadth_controls",
    }:
        return f"dffb:{raw['policy_id']}:{raw['clock']}", "entry_time_utc", "exit_time_utc", True
    if name == "sofr_rate_dislocation_primary":
        return "sfrd:SFRD-1:primary", "entry_time", "exit_time", True
    if name == "bank_deposit_secured_repo_concordance_all_clocks":
        return f"bdrc:{raw['clock_name']}", "entry_time", "exit_time", True
    if name == "fed_h8_deposit_migration_primary":
        return "h8dm:primary", "entry_time", "exit_time", raw["clock_mode"] == "primary"
    if name == "soma_lending_collateral_scarcity_primary":
        return "slcs:primary", "entry_time", "exit_time", raw["control"] == "primary"
    if name == "cross_domain_liquidity_transmission_all_clocks":
        return f"cdltr:{raw['clock']}", "entry_time_utc", "exit_time_utc", True
    if name == "live_portfolio_pure_clocks":
        return f"live:{raw['candidate_id']}", "entry_time", "exit_time", True
    raise RuntimeError(f"unknown RVFC comparator: {name}")


def load_comparator_groups() -> tuple[dict[str, list[ComparatorEvent]], int]:
    groups: dict[str, list[ComparatorEvent]] = defaultdict(list)
    rows_read = 0
    for spec in prereg.COMPARATOR_SPECS:
        name = str(spec["name"])
        if sha256_file(spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"RVFC comparator hash mismatch: {name}")
        included_for_spec = 0
        with gzip.open(_repository_path(spec["path"]), "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != COMPARATOR_HEADERS[name]:
                raise RuntimeError(f"RVFC comparator header changed: {name}")
            for raw in reader:
                rows_read += 1
                group, entry_field, exit_field, include = _comparator_identity(name, raw)
                if not include:
                    continue
                included_for_spec += 1
                event = ComparatorEvent(
                    entry_time=_timestamp(raw[entry_field]),
                    exit_time=_timestamp(raw[exit_field]),
                    side=_side(raw["side"]),
                )
                if event.exit_time <= event.entry_time:
                    raise RuntimeError("RVFC comparator interval is invalid")
                if event.entry_time >= SELECTION_END or event.exit_time > SELECTION_END:
                    raise RuntimeError("RVFC comparator opened a post-2023 clock")
                groups[group].append(event)
        if included_for_spec == 0:
            raise RuntimeError(f"RVFC required comparator is empty: {name}")
    if not groups:
        raise RuntimeError("RVFC comparator cohort is empty")
    normalized: dict[str, list[ComparatorEvent]] = {}
    for name, events in sorted(groups.items()):
        rows = sorted(events, key=lambda row: row.entry_time)
        if len({row.entry_time for row in rows}) != len(rows):
            raise RuntimeError(f"duplicate RVFC comparator entry: {name}")
        if any(current.entry_time < previous.exit_time for previous, current in zip(rows, rows[1:])):
            raise RuntimeError(f"overlapping RVFC comparator: {name}")
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
        "rvfc_one_day_containment": matches / len(a) if a else 0.0,
        "comparator_one_day_containment": matches / len(b) if b else 0.0,
        "signed_5m_occupied_exposure_correlation": correlation,
    }


def _clock_row(row: Scheduled) -> dict[str, str]:
    return {
        "control": row.control,
        "observation_date": row.observation_date.isoformat(),
        "signal_time": row.signal_time.isoformat(),
        "entry_time": row.entry_time.isoformat(),
        "exit_time": row.exit_time.isoformat(),
        "split": row.split,
        "side": str(row.side),
        "state": str(row.state),
        "score_fraction": str(row.score),
        "u_rate_dispersion": str(row.units["rate_dispersion"]),
        "u_venue_hhi": str(row.units["venue_hhi"]),
        "u_collateral_rate_disagreement": str(
            row.units["collateral_rate_disagreement"]
        ),
        "u_collateral_mix_disagreement": str(
            row.units["collateral_mix_disagreement"]
        ),
        "dominant_rate_venue": row.dominant_rate_venue,
        "dominant_volume_venue": row.dominant_volume_venue,
        "dominant_collateral_spread_venue": row.dominant_collateral_spread_venue,
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
        raise RuntimeError("RVFC output escaped repository") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_or_verify(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing RVFC artifact differs: {path.name}")
        return "verified_existing"
    _atomic_write(path, payload)
    return "created"


def _dominance(rows: Sequence[Scheduled]) -> dict[str, Any]:
    def distribution(values: Iterable[str]) -> dict[str, Any]:
        counts = Counter(values)
        total = sum(counts.values())
        return {
            "counts": dict(sorted(counts.items())),
            "shares": (
                {key: count / total for key, count in sorted(counts.items())}
                if total
                else {}
            ),
        }

    output: dict[str, Any] = {}
    for split, start, end in (
        ("train", TRAIN_START, TRAIN_END),
        ("selection", SELECTION_START, SELECTION_END),
    ):
        selected = _contained(rows, start, end)
        by_year: dict[str, Any] = {}
        for year in sorted({row.entry_time.year for row in selected}):
            year_rows = [row for row in selected if row.entry_time.year == year]
            by_year[str(year)] = {
                "events": len(year_rows),
                "dominant_rate_venue": distribution(
                    row.dominant_rate_venue for row in year_rows
                ),
                "dominant_volume_venue": distribution(
                    row.dominant_volume_venue for row in year_rows
                ),
                "dominant_collateral_spread_venue": distribution(
                    row.dominant_collateral_spread_venue for row in year_rows
                ),
            }
        output[split] = {
            "events": len(selected),
            "dominant_rate_venue": distribution(
                row.dominant_rate_venue for row in selected
            ),
            "dominant_volume_venue": distribution(
                row.dominant_volume_venue for row in selected
            ),
            "dominant_collateral_spread_venue": distribution(
                row.dominant_collateral_spread_venue for row in selected
            ),
            "by_year": by_year,
        }
    return output


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
    clocks = build_clocks(rank_rows)
    if not all(_clock_valid(name, rows) for name, rows in clocks.items()):
        raise RuntimeError("RVFC generated an invalid control clock")
    source_checks, summaries = source_support(clocks["primary"], registration["policy"])
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
        groups, comparator_rows_read = load_comparators()
        metrics = {
            name: novelty_metrics(clocks["primary"], rows)
            for name, rows in groups.items()
        }
        minimum = registration["policy"]["novelty"]["minimum_comparator_entries"]
        qualifying = sorted(
            name
            for name, row in metrics.items()
            if row["comparator_entries"] >= minimum
        )
        if not qualifying:
            raise RuntimeError("RVFC has no qualifying comparator groups")
        limits = registration["policy"]["novelty"]
        checks: dict[str, dict[str, bool]] = {}
        for name in qualifying:
            row = metrics[name]
            correlation = row["signed_5m_occupied_exposure_correlation"]
            checks[name] = {
                "exact_entry_jaccard": row["exact_entry_jaccard"]
                <= limits["maximum_exact_entry_jaccard"],
                "one_day_containment": row["rvfc_one_day_containment"]
                <= limits["maximum_rvfc_one_day_containment"],
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
        _clock_row(row)
        for control in CONTROL_NAMES
        for row in clocks[control]
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
        "rank_ready_dates": len(rank_rows),
        "clock_summaries": control_summaries,
        "primary_support_summaries": summaries,
        "source_checks": source_checks,
        "source_support_passed": source_passed,
        "dominance_diagnostics": _dominance(clocks["primary"]),
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
            "candidate_components_and_incidence_opened": True,
            "comparator_rows_read": comparator_rows_read,
            "comparator_access_short_circuited_on_source_failure": not source_passed,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
            "post_2023_source_rows_read": 0,
        },
        "advance_to_evaluator_freeze": bool(source_passed and novelty["passed"]),
        "disposition": (
            "advance unchanged to evaluator freeze"
            if source_passed and novelty["passed"]
            else "reject RVFC-72 unchanged before outcomes"
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
        (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "source_support_passed": report["source_support_passed"],
                "novelty_evaluated": report["novelty"]["evaluated"],
                "novelty_passed": report["novelty"]["passed"],
                "advance_to_evaluator_freeze": report["advance_to_evaluator_freeze"],
                "manifest_hash": report["manifest_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
