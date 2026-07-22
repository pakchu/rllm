"""Build outcome-blind SLCS-72 source, control, and novelty clocks."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from training import preregister_soma_lending_collateral_scarcity as prereg


PROTOCOL_VERSION = "soma_lending_collateral_scarcity_source_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_soma_lending_collateral_scarcity_support.py")
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "517d53437db55773bd98d7513ee5722dd1c03769a519393985f978994b3edc1a"
)
DEFAULT_OUTPUT = Path(
    "results/soma_lending_collateral_scarcity_support_2026-07-23.json"
)
DEFAULT_CLOCK = Path(
    "results/soma_lending_collateral_scarcity_clocks_2026-07-23.csv.gz"
)
HISTORY = 252
HOLD = timedelta(hours=72)
BAR = timedelta(minutes=5)
UTC = timezone.utc
COMPONENTS = (
    "demand_intensity",
    "weighted_fee",
    "carry_intensity",
    "demand_breadth",
)
SOURCE_CONTROLS = (
    "primary",
    "demand_intensity_only",
    "weighted_fee_only",
    "carry_intensity_only",
    "demand_breadth_only",
    "mean_without_consensus",
    "same_sign_without_magnitude",
    "one_operation_stale",
    "five_operation_stale",
    "year_component_permutation",
)
ECONOMIC_CONTROLS = (
    "exact_direction_flip",
    "deterministic_random_side",
    "constant_long",
    "constant_short",
)
CONTROL_NAMES = SOURCE_CONTROLS + ECONOMIC_CONTROLS
WINDOWS = {
    "2020": ("2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"),
    "2021": ("2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
    "2022": ("2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
    "train": ("2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
    "2023_h1": ("2023-01-01T00:00:00Z", "2023-07-01T00:00:00Z"),
    "2023_h2": ("2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    "selection": ("2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
}
CLOCK_COLUMNS = (
    "control",
    "operation_id",
    "operation_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "split",
    "side",
    "state",
    "score",
    "u_demand_intensity",
    "u_weighted_fee",
    "u_carry_intensity",
    "u_demand_breadth",
)


@dataclass(frozen=True)
class OperationRow:
    operation_id: str
    operation_date: str
    available_at: datetime
    total_submitted: Decimal
    total_accepted: Decimal


@dataclass(frozen=True)
class DetailRow:
    operation_id: str
    operation_date: str
    available_at: datetime
    cusip: str
    submitted: Decimal
    accepted: Decimal
    weighted_rate: Decimal | None
    actual_available: Decimal
    outstanding_loans: Decimal


@dataclass(frozen=True)
class FeatureRow:
    operation_id: str
    operation_date: str
    available_at: datetime
    complete: bool
    demand_intensity: Decimal | None
    weighted_fee: Decimal | None
    carry_intensity: Decimal | None
    demand_breadth: Decimal | None


@dataclass(frozen=True)
class StateRow:
    operation_id: str
    operation_date: str
    available_at: datetime
    rank_ready: bool
    state: int
    score: float
    u_demand_intensity: float | None
    u_weighted_fee: float | None
    u_carry_intensity: float | None
    u_demand_breadth: float | None


@dataclass(frozen=True)
class Candidate:
    operation_id: str
    operation_date: str
    signal_time: datetime
    side: int
    state: int
    score: float
    u_demand_intensity: float | None
    u_weighted_fee: float | None
    u_carry_intensity: float | None
    u_demand_breadth: float | None


@dataclass(frozen=True)
class Scheduled:
    control: str
    operation_id: str
    operation_date: str
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    split: str
    side: int
    state: int
    score: float
    u_demand_intensity: float | None
    u_weighted_fee: float | None
    u_carry_intensity: float | None
    u_demand_breadth: float | None


@dataclass(frozen=True)
class ComparatorEvent:
    entry_time: datetime
    exit_time: datetime
    side: int


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("path must remain repository-relative") from exc
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any) -> datetime:
    parsed = pd.Timestamp(value)
    if parsed is pd.NaT:
        raise RuntimeError("SLCS timestamp is NaT")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.to_pydatetime()


def _decimal(value: Any, label: str, *, optional: bool = False) -> Decimal | None:
    if optional and (value is None or str(value) == ""):
        return None
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise RuntimeError(f"SLCS {label} must be finite and nonnegative")
    return result


def _load_registration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("SLCS preregistration file hash mismatch")
    payload = json.loads(_repository_path(PREREGISTRATION).read_text())
    prereg.validate_preregistration(payload)
    if payload.get("exact_source_incidence_opened") is not False:
        raise RuntimeError("SLCS preregistration opened incidence")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("SLCS preregistration opened outcomes")
    return payload


def load_source() -> tuple[list[OperationRow], list[DetailRow]]:
    if sha256_file(prereg.OPERATIONS) != prereg.OPERATIONS_SHA256:
        raise RuntimeError("SLCS operation panel hash mismatch")
    if sha256_file(prereg.DETAILS) != prereg.DETAILS_SHA256:
        raise RuntimeError("SLCS detail panel hash mismatch")
    operations: list[OperationRow] = []
    with gzip.open(_repository_path(prereg.OPERATIONS), "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "operation_id",
            "operation_date",
            "available_at_utc",
            "total_par_submitted",
            "total_par_accepted",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise RuntimeError("SLCS operation panel schema changed")
        for raw in reader:
            submitted = _decimal(raw["total_par_submitted"], "operation submitted")
            accepted = _decimal(raw["total_par_accepted"], "operation accepted")
            assert isinstance(submitted, Decimal)
            assert isinstance(accepted, Decimal)
            operations.append(
                OperationRow(
                    operation_id=raw["operation_id"],
                    operation_date=raw["operation_date"],
                    available_at=_timestamp(raw["available_at_utc"]),
                    total_submitted=submitted,
                    total_accepted=accepted,
                )
            )
    details: list[DetailRow] = []
    with gzip.open(_repository_path(prereg.DETAILS), "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "operation_id",
            "operation_date",
            "available_at_utc",
            "cusip",
            "par_submitted",
            "par_accepted",
            "weighted_average_rate",
            "actual_available_to_borrow",
            "outstanding_loans",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise RuntimeError("SLCS detail panel schema changed")
        for raw in reader:
            submitted = _decimal(raw["par_submitted"], "detail submitted")
            accepted = _decimal(raw["par_accepted"], "detail accepted")
            actual = _decimal(raw["actual_available_to_borrow"], "detail available")
            outstanding = _decimal(raw["outstanding_loans"], "detail outstanding")
            rate = _decimal(raw["weighted_average_rate"], "detail rate", optional=True)
            assert isinstance(submitted, Decimal)
            assert isinstance(accepted, Decimal)
            assert isinstance(actual, Decimal)
            assert isinstance(outstanding, Decimal)
            if accepted > 0 and rate is None:
                raise RuntimeError("SLCS nonzero award is missing rate")
            details.append(
                DetailRow(
                    operation_id=raw["operation_id"],
                    operation_date=raw["operation_date"],
                    available_at=_timestamp(raw["available_at_utc"]),
                    cusip=raw["cusip"],
                    submitted=submitted,
                    accepted=accepted,
                    weighted_rate=rate,
                    actual_available=actual,
                    outstanding_loans=outstanding,
                )
            )
    if len(operations) != 1259 or len(details) != 182616:
        raise RuntimeError("SLCS source row counts changed")
    if len({row.operation_id for row in operations}) != len(operations):
        raise RuntimeError("SLCS operation identity duplicated")
    if len({(row.operation_id, row.cusip) for row in details}) != len(details):
        raise RuntimeError("SLCS detail identity duplicated")
    if any(
        current.available_at < previous.available_at
        for previous, current in zip(operations, operations[1:])
    ):
        raise RuntimeError("SLCS source availability is not nondecreasing")
    return operations, details


def build_features(
    operations: Sequence[OperationRow], details: Sequence[DetailRow]
) -> list[FeatureRow]:
    grouped: dict[str, list[DetailRow]] = defaultdict(list)
    for row in details:
        grouped[row.operation_id].append(row)
    features: list[FeatureRow] = []
    for operation in operations:
        rows = grouped.get(operation.operation_id, [])
        if not rows:
            features.append(
                FeatureRow(
                    operation.operation_id,
                    operation.operation_date,
                    operation.available_at,
                    False,
                    None,
                    None,
                    None,
                    None,
                )
            )
            continue
        if any(
            row.operation_date != operation.operation_date
            or row.available_at != operation.available_at
            for row in rows
        ):
            raise RuntimeError("SLCS operation/detail join identity mismatch")
        submitted = sum((row.submitted for row in rows), Decimal(0))
        accepted = sum((row.accepted for row in rows), Decimal(0))
        if submitted != operation.total_submitted or accepted != operation.total_accepted:
            raise RuntimeError("SLCS operation/detail totals stopped reconciling")
        available = sum((row.actual_available for row in rows), Decimal(0))
        outstanding = sum((row.outstanding_loans for row in rows), Decimal(0))
        available_cusips = sum(row.actual_available > 0 for row in rows)
        submitted_cusips = sum(row.submitted > 0 for row in rows)
        weighted_numerator = sum(
            (
                row.accepted * row.weighted_rate
                for row in rows
                if row.accepted > 0 and row.weighted_rate is not None
            ),
            Decimal(0),
        )
        complete = available > 0 and accepted > 0 and available_cusips > 0
        features.append(
            FeatureRow(
                operation_id=operation.operation_id,
                operation_date=operation.operation_date,
                available_at=operation.available_at,
                complete=complete,
                demand_intensity=(submitted / available if complete else None),
                weighted_fee=(weighted_numerator / accepted if complete else None),
                carry_intensity=(outstanding / available if complete else None),
                demand_breadth=(
                    Decimal(submitted_cusips) / Decimal(available_cusips)
                    if complete
                    else None
                ),
            )
        )
    return features


def midrank_unit(current: Decimal, prior: Sequence[Decimal]) -> float:
    if len(prior) != HISTORY:
        raise RuntimeError("SLCS midrank history length changed")
    lower = sum(value < current for value in prior)
    equal = sum(value == current for value in prior)
    rank = (Decimal(lower) + Decimal("0.5") * Decimal(equal)) / Decimal(HISTORY)
    return float(Decimal(2) * rank - Decimal(1))


def primary_state(vector: Mapping[str, float]) -> tuple[int, float]:
    values = [vector[name] for name in COMPONENTS]
    score = float(sum(values) / len(values))
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    if positive >= 3 and score >= 0.50:
        return 1, score
    if negative >= 3 and score <= -0.50:
        return -1, score
    return 0, score


def component_state(value: float) -> tuple[int, float]:
    if value >= 0.50:
        return 1, value
    if value <= -0.50:
        return -1, value
    return 0, value


def sign_consensus_state(vector: Mapping[str, float]) -> tuple[int, float]:
    values = [vector[name] for name in COMPONENTS]
    score = float(sum(values) / len(values))
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    if positive >= 3:
        return 1, score
    if negative >= 3:
        return -1, score
    return 0, score


def mean_state(vector: Mapping[str, float]) -> tuple[int, float]:
    score = float(sum(vector[name] for name in COMPONENTS) / len(COMPONENTS))
    if score >= 0.50:
        return 1, score
    if score <= -0.50:
        return -1, score
    return 0, score


def build_rank_rows(features: Sequence[FeatureRow]) -> list[StateRow]:
    histories: dict[str, list[Decimal]] = {name: [] for name in COMPONENTS}
    rows: list[StateRow] = []
    index = 0
    while index < len(features):
        batch_end = index + 1
        while (
            batch_end < len(features)
            and features[batch_end].available_at == features[index].available_at
        ):
            batch_end += 1
        batch = features[index:batch_end]
        complete_values: list[dict[str, Decimal]] = []
        for feature in batch:
            raw_values = {name: getattr(feature, name) for name in COMPONENTS}
            complete = feature.complete and all(
                isinstance(value, Decimal) for value in raw_values.values()
            )
            if not complete:
                rows.append(
                    StateRow(
                        feature.operation_id,
                        feature.operation_date,
                        feature.available_at,
                        False,
                        0,
                        0.0,
                        None,
                        None,
                        None,
                        None,
                    )
                )
                continue
            values = {
                name: value
                for name, value in raw_values.items()
                if isinstance(value, Decimal)
            }
            if len(values) != len(COMPONENTS):
                raise RuntimeError("SLCS complete row lost a component")
            ready = all(len(histories[name]) >= HISTORY for name in COMPONENTS)
            if ready:
                vector = {
                    name: midrank_unit(values[name], histories[name][-HISTORY:])
                    for name in COMPONENTS
                }
                state, score = primary_state(vector)
                rows.append(
                    StateRow(
                        feature.operation_id,
                        feature.operation_date,
                        feature.available_at,
                        True,
                        state,
                        score,
                        vector["demand_intensity"],
                        vector["weighted_fee"],
                        vector["carry_intensity"],
                        vector["demand_breadth"],
                    )
                )
            else:
                rows.append(
                    StateRow(
                        feature.operation_id,
                        feature.operation_date,
                        feature.available_at,
                        False,
                        0,
                        0.0,
                        None,
                        None,
                        None,
                        None,
                    )
                )
            complete_values.append(values)
        for values in complete_values:
            for name in COMPONENTS:
                histories[name].append(values[name])
        index = batch_end
    return rows


def _vector(row: StateRow) -> dict[str, float] | None:
    values = {
        "demand_intensity": row.u_demand_intensity,
        "weighted_fee": row.u_weighted_fee,
        "carry_intensity": row.u_carry_intensity,
        "demand_breadth": row.u_demand_breadth,
    }
    if not row.rank_ready or any(value is None for value in values.values()):
        return None
    return {name: float(value) for name, value in values.items()}


def control_state_rows(rows: Sequence[StateRow], control: str) -> list[StateRow]:
    if control == "primary":
        return list(rows)
    if control in {f"{name}_only" for name in COMPONENTS}:
        component = control.removesuffix("_only")
        output = []
        for row in rows:
            vector = _vector(row)
            if vector is None:
                output.append(replace(row, rank_ready=False, state=0, score=0.0))
                continue
            state, score = component_state(vector[component])
            output.append(replace(row, state=state, score=score))
        return output
    if control in {"mean_without_consensus", "same_sign_without_magnitude"}:
        classifier = mean_state if control == "mean_without_consensus" else sign_consensus_state
        output = []
        for row in rows:
            vector = _vector(row)
            if vector is None:
                output.append(replace(row, rank_ready=False, state=0, score=0.0))
                continue
            state, score = classifier(vector)
            output.append(replace(row, state=state, score=score))
        return output
    if control in {"one_operation_stale", "five_operation_stale"}:
        lag = 1 if control == "one_operation_stale" else 5
        output = []
        for index, row in enumerate(rows):
            if index < lag or any(
                _vector(item) is None for item in rows[index - lag : index + 1]
            ):
                output.append(replace(row, rank_ready=False, state=0, score=0.0))
                continue
            vector = _vector(rows[index - lag])
            assert vector is not None
            state, score = primary_state(vector)
            output.append(
                replace(
                    row,
                    rank_ready=True,
                    state=state,
                    score=score,
                    u_demand_intensity=vector["demand_intensity"],
                    u_weighted_fee=vector["weighted_fee"],
                    u_carry_intensity=vector["carry_intensity"],
                    u_demand_breadth=vector["demand_breadth"],
                )
            )
        return output
    if control == "year_component_permutation":
        return permuted_state_rows(rows)
    raise RuntimeError(f"unknown SLCS source control: {control}")


def _permutation_order(year: str, component: str, operation_id: str) -> bytes:
    return hashlib.sha256(
        f"SLCS-72|{year}|{component}|{operation_id}".encode("utf-8")
    ).digest()


def permuted_state_rows(rows: Sequence[StateRow]) -> list[StateRow]:
    assigned: dict[tuple[int, str], float] = {}
    by_year: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if _vector(row) is not None:
            by_year[row.operation_date[:4]].append(index)
    for year, indices in by_year.items():
        for component in COMPONENTS:
            source_values = [getattr(rows[index], f"u_{component}") for index in indices]
            targets = sorted(
                indices,
                key=lambda index: _permutation_order(
                    year, component, rows[index].operation_id
                ),
            )
            for target, value in zip(targets, source_values):
                assert value is not None
                assigned[(target, component)] = float(value)
    output = []
    for index, row in enumerate(rows):
        if _vector(row) is None:
            output.append(replace(row, rank_ready=False, state=0, score=0.0))
            continue
        vector = {name: assigned[(index, name)] for name in COMPONENTS}
        state, score = primary_state(vector)
        output.append(
            replace(
                row,
                state=state,
                score=score,
                u_demand_intensity=vector["demand_intensity"],
                u_weighted_fee=vector["weighted_fee"],
                u_carry_intensity=vector["carry_intensity"],
                u_demand_breadth=vector["demand_breadth"],
            )
        )
    return output


def candidates_from_states(rows: Sequence[StateRow]) -> list[Candidate]:
    candidates: list[Candidate] = []
    previous: StateRow | None = None
    for row in rows:
        if not row.rank_ready:
            previous = None
            continue
        if previous is not None and row.state in (-1, 1) and row.state != previous.state:
            candidates.append(
                Candidate(
                    operation_id=row.operation_id,
                    operation_date=row.operation_date,
                    signal_time=row.available_at,
                    side=-row.state,
                    state=row.state,
                    score=row.score,
                    u_demand_intensity=row.u_demand_intensity,
                    u_weighted_fee=row.u_weighted_fee,
                    u_carry_intensity=row.u_carry_intensity,
                    u_demand_breadth=row.u_demand_breadth,
                )
            )
        previous = row
    return candidates


def _ceil_5m(value: datetime) -> datetime:
    value = value.astimezone(UTC)
    floored = value.replace(second=0, microsecond=0, minute=(value.minute // 5) * 5)
    return floored if floored == value else floored + BAR


def _split(entry: datetime, exit_time: datetime) -> str | None:
    for name in ("train", "selection"):
        start, end = (_timestamp(item) for item in WINDOWS[name])
        if entry >= start and exit_time <= end:
            return name
    return None


def schedule(control: str, candidates: Sequence[Candidate]) -> list[Scheduled]:
    accepted: list[Scheduled] = []
    prior_exit: datetime | None = None
    for candidate in sorted(candidates, key=lambda row: row.signal_time):
        entry = _ceil_5m(candidate.signal_time) + BAR
        exit_time = entry + HOLD
        split = _split(entry, exit_time)
        if split is None:
            continue
        if prior_exit is not None and entry < prior_exit:
            continue
        accepted.append(
            Scheduled(
                control=control,
                operation_id=candidate.operation_id,
                operation_date=candidate.operation_date,
                signal_time=candidate.signal_time,
                entry_time=entry,
                exit_time=exit_time,
                split=split,
                side=candidate.side,
                state=candidate.state,
                score=candidate.score,
                u_demand_intensity=candidate.u_demand_intensity,
                u_weighted_fee=candidate.u_weighted_fee,
                u_carry_intensity=candidate.u_carry_intensity,
                u_demand_breadth=candidate.u_demand_breadth,
            )
        )
        prior_exit = exit_time
    return accepted


def _random_side(operation_id: str, entry_time: datetime) -> int:
    digest = hashlib.sha256(
        f"SLCS-72-random|{operation_id}|{entry_time.isoformat()}".encode()
    ).digest()
    return 1 if digest[0] < 128 else -1


def _side(value: Any) -> int:
    normalized = str(value).strip().upper()
    mapping = {"1": 1, "+1": 1, "LONG": 1, "-1": -1, "SHORT": -1}
    if normalized not in mapping:
        raise RuntimeError(f"unknown SLCS comparator side: {value!r}")
    return mapping[normalized]


def build_control_clocks(rank_rows: Sequence[StateRow]) -> dict[str, list[Scheduled]]:
    clocks = {
        control: schedule(
            control,
            candidates_from_states(control_state_rows(rank_rows, control)),
        )
        for control in SOURCE_CONTROLS
    }
    primary = clocks["primary"]
    clocks["exact_direction_flip"] = [
        replace(row, control="exact_direction_flip", side=-row.side)
        for row in primary
    ]
    clocks["deterministic_random_side"] = [
        replace(
            row,
            control="deterministic_random_side",
            side=_random_side(row.operation_id, row.entry_time),
        )
        for row in primary
    ]
    clocks["constant_long"] = [
        replace(row, control="constant_long", side=1) for row in primary
    ]
    clocks["constant_short"] = [
        replace(row, control="constant_short", side=-1) for row in primary
    ]
    return clocks


def _contained(
    rows: Iterable[Scheduled], start: str | datetime, end: str | datetime
) -> list[Scheduled]:
    lower = _timestamp(start)
    upper = _timestamp(end)
    return [row for row in rows if row.entry_time >= lower and row.exit_time <= upper]


def _summary(rows: Sequence[Scheduled], start: str, end: str) -> dict[str, Any]:
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
    operation_ids = [row.operation_id for row in rows]
    entry_times = [row.entry_time for row in rows]
    return (
        len(operation_ids) == len(set(operation_ids))
        and len(entry_times) == len(set(entry_times))
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


def _comparator_groups() -> dict[str, list[ComparatorEvent]]:
    groups: dict[str, list[ComparatorEvent]] = defaultdict(list)
    for spec in prereg.COMPARATOR_SPECS:
        path = _repository_path(spec["path"])
        if sha256_file(spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"SLCS comparator hash mismatch: {spec['name']}")
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
        name = str(spec["name"])
        for raw in rows:
            if name == "federal_liquidity_component_concordance":
                group = f"{name}:{raw['candidate_id']}:{raw['clock_name']}"
                exit_field = "exit_time"
            elif name == "overnight_rrp_flow_release":
                if raw["clock_mode"] != "primary":
                    continue
                group = name
                exit_field = "scheduled_exit_time"
            elif name == "sofr_rate_dislocation":
                group = name
                exit_field = "exit_time"
            elif name == "fed_h8_deposit_migration":
                if raw["clock_mode"] != "primary":
                    continue
                group = name
                exit_field = "exit_time"
            elif name == "live_portfolio_pure_clocks":
                group = f"{name}:{raw['candidate_id']}"
                exit_field = "exit_time"
            else:
                raise RuntimeError(f"unknown SLCS comparator: {name}")
            groups[group].append(
                ComparatorEvent(
                    entry_time=_timestamp(raw["entry_time"]),
                    exit_time=_timestamp(raw[exit_field]),
                    side=_side(raw["side"]),
                )
            )
    return {name: sorted(rows, key=lambda row: row.entry_time) for name, rows in groups.items()}


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
    start = _timestamp("2020-01-01T00:00:00Z")
    end = _timestamp("2024-01-01T00:00:00Z")
    primary_events = [
        ComparatorEvent(row.entry_time, row.exit_time, row.side)
        for row in primary
        if row.entry_time >= start and row.exit_time <= end
    ]
    comparator_events = [
        row for row in comparator if row.entry_time >= start and row.exit_time <= end
    ]
    a = [row.entry_time for row in primary_events]
    b = [row.entry_time for row in comparator_events]
    exact_a = set(a)
    exact_b = set(b)
    union = exact_a | exact_b
    matches = one_to_one_matches(a, b, timedelta(hours=24))
    left_exposure = _exposure(primary_events, start, end)
    right_exposure = _exposure(comparator_events, start, end)
    if np.std(left_exposure) == 0 or np.std(right_exposure) == 0:
        correlation: float | None = None
    else:
        correlation = float(np.corrcoef(left_exposure, right_exposure)[0, 1])
    return {
        "primary_entries": len(a),
        "comparator_entries": len(b),
        "exact_entry_intersection": len(exact_a & exact_b),
        "exact_entry_jaccard": len(exact_a & exact_b) / len(union) if union else 0.0,
        "one_day_one_to_one_matches": matches,
        "slcs_one_day_containment": matches / len(a) if a else 0.0,
        "comparator_one_day_containment": matches / len(b) if b else 0.0,
        "one_day_one_to_one_jaccard": (
            matches / (len(a) + len(b) - matches)
            if len(a) + len(b) - matches
            else 0.0
        ),
        "signed_5m_occupied_exposure_correlation": correlation,
    }


def _clock_row(row: Scheduled) -> dict[str, Any]:
    payload = asdict(row)
    return {
        "control": payload["control"],
        "operation_id": payload["operation_id"],
        "operation_date": payload["operation_date"],
        "signal_time": payload["signal_time"].isoformat(),
        "entry_time": payload["entry_time"].isoformat(),
        "exit_time": payload["exit_time"].isoformat(),
        "split": payload["split"],
        "side": payload["side"],
        "state": payload["state"],
        "score": payload["score"],
        "u_demand_intensity": payload["u_demand_intensity"],
        "u_weighted_fee": payload["u_weighted_fee"],
        "u_carry_intensity": payload["u_carry_intensity"],
        "u_demand_breadth": payload["u_demand_breadth"],
    }


def _write_clock(path: Path, clocks: Mapping[str, Sequence[Scheduled]]) -> None:
    rows = [
        _clock_row(row)
        for control in CONTROL_NAMES
        for row in clocks[control]
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(CLOCK_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(buffer.getvalue().encode(), compresslevel=9, mtime=0))


def build_report(
    *, clock_output: str | Path = DEFAULT_CLOCK, write_clock: bool = True
) -> dict[str, Any]:
    registration = _load_registration()
    operations, details = load_source()
    features = build_features(operations, details)
    rank_rows = build_rank_rows(features)
    clocks = build_control_clocks(rank_rows)
    summaries = {
        control: {
            name: _summary(rows, *window) for name, window in WINDOWS.items()
        }
        for control, rows in clocks.items()
    }
    primary = summaries["primary"]
    gates = {
        "train_total_minimum_60": primary["train"]["events"] >= 60,
        "each_train_year_minimum_15": all(
            primary[year]["events"] >= 15 for year in ("2020", "2021", "2022")
        ),
        "train_each_side_minimum_15": min(
            primary["train"]["longs"], primary["train"]["shorts"]
        ) >= 15,
        "selection_total_minimum_18": primary["selection"]["events"] >= 18,
        "each_selection_half_minimum_7": all(
            primary[half]["events"] >= 7 for half in ("2023_h1", "2023_h2")
        ),
        "selection_each_side_minimum_4": min(
            primary["selection"]["longs"], primary["selection"]["shorts"]
        ) >= 4,
        "every_train_quarter_active": primary["train"]["active_quarters"] == 12,
        "every_selection_quarter_active": primary["selection"]["active_quarters"] == 4,
        "train_maximum_month_share_15pct": primary["train"]["max_single_month_share"] <= 0.15,
        "selection_maximum_month_share_20pct": primary["selection"]["max_single_month_share"] <= 0.20,
        "train_maximum_gap_45d": primary["train"]["maximum_entry_gap_elapsed_days"] <= 45,
        "selection_maximum_gap_45d": primary["selection"]["maximum_entry_gap_elapsed_days"] <= 45,
    }
    comparator_groups = _comparator_groups()
    novelty = {
        name: novelty_metrics(clocks["primary"], rows)
        for name, rows in comparator_groups.items()
    }
    qualifying = {
        name: values
        for name, values in novelty.items()
        if values["comparator_entries"] >= 10
    }
    novelty_checks = {
        name: {
            "exact_entry_jaccard_at_most_0_10": values["exact_entry_jaccard"] <= 0.10,
            "slcs_one_day_containment_at_most_0_35": values["slcs_one_day_containment"] <= 0.35,
            "absolute_signed_exposure_correlation_at_most_0_35": (
                values["signed_5m_occupied_exposure_correlation"] is not None
                and abs(values["signed_5m_occupied_exposure_correlation"]) <= 0.35
            ),
        }
        for name, values in qualifying.items()
    }
    source_checks = {
        "source_hashes_match": True,
        "operation_rows_exactly_1259": len(operations) == 1259,
        "detail_rows_exactly_182616": len(details) == 182616,
        "all_operations_feature_complete": all(row.complete for row in features),
        "rank_history_exactly_252": HISTORY == 252,
        "all_controls_present": set(clocks) == set(CONTROL_NAMES),
        "all_clocks_valid": all(
            _clock_valid(control, rows) for control, rows in clocks.items()
        ),
        "flip_uses_exact_primary_clock": all(
            original.entry_time == flipped.entry_time and original.side == -flipped.side
            for original, flipped in zip(
                clocks["primary"], clocks["exact_direction_flip"]
            )
        ),
        "constant_controls_use_exact_primary_clock": (
            len(clocks["constant_long"]) == len(clocks["primary"])
            and len(clocks["constant_short"]) == len(clocks["primary"])
        ),
        "market_funding_return_rows_opened_zero": True,
    }
    support_passed = all(source_checks.values()) and all(gates.values())
    novelty_passed = bool(qualifying) and all(
        all(checks.values()) for checks in novelty_checks.values()
    )
    clock_path = _repository_path(clock_output)
    if write_clock:
        _write_clock(clock_path, clocks)
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": prereg.POLICY_ID,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "preregistration_file_sha256": PREREGISTRATION_SHA256,
        "support_builder": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "source": {
            "operation_rows": len(operations),
            "detail_rows": len(details),
            "feature_complete_rows": sum(row.complete for row in features),
            "rank_ready_rows": sum(row.rank_ready for row in rank_rows),
            "first_rank_ready_operation": next(
                (row.operation_id for row in rank_rows if row.rank_ready), None
            ),
        },
        "clock_summaries": summaries,
        "source_checks": source_checks,
        "source_support_gates": gates,
        "source_support_passed": support_passed,
        "novelty_metrics": novelty,
        "novelty_checks": novelty_checks,
        "qualifying_comparator_groups": len(qualifying),
        "novelty_passed": novelty_passed,
        "advance_to_evaluator_freeze": support_passed and novelty_passed,
        "outcome_boundary": {
            "source_operation_rows_read": len(operations),
            "source_detail_rows_read": len(details),
            "comparator_clock_rows_read": sum(len(rows) for rows in comparator_groups.values()),
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
            "economic_simulations_run": 0,
        },
        "clocks": {
            "path": str(Path(clock_output)),
            "sha256": sha256_file(Path(clock_output)) if write_clock else None,
            "rows": sum(len(rows) for rows in clocks.values()),
            "columns": list(CLOCK_COLUMNS),
        },
        "disposition": (
            "FREEZE_EVALUATOR"
            if support_passed and novelty_passed
            else "REJECT_BEFORE_OUTCOMES_NO_REPAIR"
        ),
    }
    core["manifest_hash"] = canonical_hash(core)
    return core


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK))
    args = parser.parse_args(argv)
    report = build_report(clock_output=args.clock_output)
    output = _repository_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "disposition": report["disposition"],
                "source_support_passed": report["source_support_passed"],
                "novelty_passed": report["novelty_passed"],
                "train_events": report["clock_summaries"]["primary"]["train"]["events"],
                "selection_events": report["clock_summaries"]["primary"]["selection"]["events"],
                "outcomes_opened": False,
                "manifest_hash": report["manifest_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
