"""Build outcome-blind SMAF-72 source-support and novelty evidence."""

from __future__ import annotations

import argparse
import csv
import errno
import gzip
import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import (
    preregister_soma_maturity_allocation_fracture as prereg,
)

PROTOCOL_VERSION = "soma_maturity_allocation_fracture_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/build_soma_maturity_allocation_fracture_support.py"
)
TEST_PATH = Path(
    "tests/test_build_soma_maturity_allocation_fracture_support.py"
)
IMPLEMENTATION_CONTRACT = Path(
    "docs/smaf-source-support-implementation-contract-2026-07-30.md"
)
IMPLEMENTATION_CONTRACT_SHA256 = (
    "9238b64b7651fd21727fccb5922cc14d527c72f290e57428c1b949ce7bd6a63c"
)

PREREGISTRATION_DOCUMENT = prereg.PREREGISTRATION_DOCUMENT
PREREGISTRATION_DOCUMENT_SHA256 = (
    "0ca0b00c77bd55e3360abe1f36409938a8e95dc450a599b89370d1265b4491f9"
)
PREREGISTRATION_BUILDER = prereg.SCRIPT_PATH
PREREGISTRATION_BUILDER_SHA256 = (
    "1712afb46872a3b94cc9838b9225c596d2c19b6d4ec5da672a4e9d562e73e554"
)
PREREGISTRATION_TEST = Path(
    "tests/test_preregister_soma_maturity_allocation_fracture.py"
)
PREREGISTRATION_TEST_SHA256 = (
    "d68e80cd6bfa4c49d8d98698c2930b72fd2949e21ba3d27d59d5098ef559fb88"
)
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "4809586182ab2777d1aa32cc249e223700419983fb85a3d149532c62c0e0d01d"
)
PREREGISTRATION_MANIFEST_HASH = (
    "a5949fd0aa723aebf271a966371222c219093fbff1f3d34ba383f5f66620682b"
)
PREREGISTRATION_COMMIT = "01bdc8b923f1ddd4e218df28239e2b814fc47f62"

DEFAULT_CLOCK_OUTPUT = Path(
    "data/soma_maturity_allocation_fracture_clocks_2020_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/soma_maturity_allocation_fracture_support_2026-07-30.json"
)

UTC = timezone.utc
BAR_SECONDS = 300
HOLD_SECONDS = 72 * 60 * 60
COMMON_START = datetime(2020, 1, 1, tzinfo=UTC)
TRAIN_END = datetime(2023, 1, 1, tzinfo=UTC)
COMMON_END = datetime(2024, 1, 1, tzinfo=UTC)
WARMUP_START = datetime(2019, 1, 1, tzinfo=UTC)
RFC3339_UTC_SECONDS = re.compile(
    r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:Z|\+00:00)\Z",
    flags=re.ASCII,
)
ISO_DATE = re.compile(
    r"\A[0-9]{4}-(?:0[1-9]|1[0-2])-"
    r"(?:0[1-9]|[12][0-9]|3[01])\Z",
    flags=re.ASCII,
)

CLOCK_COLUMNS = (
    "control",
    "signal_id",
    "parent_signal_id",
    "decision_time",
    "entry_time",
    "exit_time",
    "split",
    "side",
    "tail",
)
CONTROL_ORDER = prereg.SOURCE_CONTROL_ORDER + prereg.OUTCOME_CONTROL_ORDER
FORBIDDEN_CLOCK_TOKENS = (
    "amount",
    "centroid",
    "fracture",
    "rank_numerator",
    "price",
    "funding",
    "return",
    "pnl",
    "cagr",
    "mdd",
    "reward",
    "label",
    "cusip",
    "operation_id",
)
SCHEMA_INVALID_REASON_CODES = (
    "invalid_operation_total_submitted",
    "invalid_operation_total_accepted",
    "operation_total_accepted_above_submitted",
    "detail_join_value_mismatch",
    "invalid_detail_par_submitted",
    "invalid_detail_par_accepted",
    "invalid_detail_actual_available_to_borrow",
    "accepted_above_submitted",
    "operation_total_reconciliation",
)
PARSER_COMPLETE_INVALID_REASON_CODES = (
    "security_description_parser",
    "operation_without_details",
    "nonpositive_centroid_weight",
)
SOURCE_STAGE_ORDER = (
    "frozen_identity_and_exact_header",
    "schema_join_uniqueness_reconciliation",
    "parser_coverage_and_complete_operations",
    "singleton_causal_batches",
    "rank_coverage_and_tail_selectivity",
    "primary_event_support",
    "internal_component_distinctness",
)


@dataclass(frozen=True)
class Operation:
    operation_id: str
    operation_date: date
    operation_date_text: str
    available_at: datetime
    available_at_text: str
    total_submitted: Fraction | None
    total_accepted: Fraction | None
    invalid_reasons: tuple[str, ...] = ()


@dataclass
class Accumulator:
    detail_rows: int = 0
    submitted: Fraction = Fraction(0)
    accepted: Fraction = Fraction(0)
    available: Fraction = Fraction(0)
    submitted_tau: Fraction = Fraction(0)
    accepted_tau: Fraction = Fraction(0)
    available_tau: Fraction = Fraction(0)


@dataclass(frozen=True)
class OperationFeature:
    operation_id: str
    operation_date: date
    available_at: datetime
    available_at_text: str
    primary: Fraction | None
    submitted_inventory_tilt: Fraction | None
    submitted_award_tilt: Fraction | None
    award_inventory_tilt: Fraction | None
    aggregate_demand_intensity: Fraction | None
    detail_rows: int
    valid: bool = True
    invalid_reasons: tuple[str, ...] = ()
    segment: int = -1

    def value(self, control: str) -> Fraction:
        if control not in prereg.SOURCE_CONTROL_ORDER:
            raise RuntimeError(f"SMAF-72 unknown source control: {control}")
        value = getattr(self, control)
        if value is None:
            raise RuntimeError("SMAF-72 invalid operation has no feature")
        return value


@dataclass(frozen=True)
class Signal:
    control: str
    signal_id: str
    parent_signal_id: str
    operation_id: str
    operation_date: date
    decision_time: datetime
    segment: int
    side: str
    tail: str
    entry_override: datetime | None = None


@dataclass(frozen=True)
class Scheduled:
    control: str
    signal_id: str
    parent_signal_id: str
    operation_id: str
    operation_date: date
    decision_time: datetime
    entry_time: datetime
    exit_time: datetime
    segment: int
    side: str
    tail: str
    split: str | None


@dataclass(frozen=True)
class SourceBuild:
    operations: tuple[Operation, ...]
    features: tuple[OperationFeature, ...]
    detail_rows: int
    valid_description_rows: int
    complete_operations: int
    batches: int
    singleton_batches: int
    invalid_reason_counts: Mapping[str, int]
    detail_rows_by_window: Mapping[str, int]
    valid_description_rows_by_window: Mapping[str, int]
    operations_by_window: Mapping[str, int]
    complete_operations_by_window: Mapping[str, int]
    batches_by_window: Mapping[str, int]
    singleton_batches_by_window: Mapping[str, int]


class SourceContractFailure(RuntimeError):
    """Carry fail-closed source evidence without opening outcomes."""

    def __init__(
        self,
        stage: str,
        code: str,
        rows_decoded: int,
        message: str,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.rows_decoded = int(rows_decoded)


class ComparatorContractFailure(RuntimeError):
    """Carry comparator failure evidence into a terminal report."""

    def __init__(self, code: str, rows_decoded: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.rows_decoded = int(rows_decoded)


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_time(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise RuntimeError("SMAF-72 timestamp is not UTC")
    if value.microsecond:
        raise RuntimeError("SMAF-72 timestamp has fractional seconds")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp(
    value: str,
    label: str,
    *,
    rows_decoded: int = 0,
) -> datetime:
    if RFC3339_UTC_SECONDS.fullmatch(value) is None:
        raise SourceContractFailure(
            "schema_join_uniqueness_reconciliation",
            "noncanonical_timestamp",
            rows_decoded,
            f"SMAF-72 noncanonical timestamp: {label}",
        )
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise SourceContractFailure(
            "schema_join_uniqueness_reconciliation",
            "invalid_timestamp",
            rows_decoded,
            f"SMAF-72 invalid timestamp: {label}",
        ) from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timedelta(0)
        or parsed.microsecond
    ):
        raise SourceContractFailure(
            "schema_join_uniqueness_reconciliation",
            "timestamp_not_utc_second",
            rows_decoded,
            f"SMAF-72 timestamp not a UTC second: {label}",
        )
    return parsed.astimezone(UTC)


def _parse_date(
    value: str,
    label: str,
    *,
    rows_decoded: int = 0,
) -> date:
    if ISO_DATE.fullmatch(value) is None:
        raise SourceContractFailure(
            "schema_join_uniqueness_reconciliation",
            "noncanonical_date",
            rows_decoded,
            f"SMAF-72 noncanonical date: {label}",
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise SourceContractFailure(
            "schema_join_uniqueness_reconciliation",
            "invalid_date",
            rows_decoded,
            f"SMAF-72 invalid date: {label}",
        ) from error
    if parsed.isoformat() != value:
        raise SourceContractFailure(
            "schema_join_uniqueness_reconciliation",
            "noncanonical_date",
            rows_decoded,
            f"SMAF-72 noncanonical date: {label}",
        )
    return parsed


def _exact_decimal(
    value: str,
    label: str,
    *,
    rows_decoded: int = 0,
) -> Fraction:
    try:
        return prereg.parse_exact_decimal(value)
    except ValueError as error:
        raise SourceContractFailure(
            "schema_join_uniqueness_reconciliation",
            "invalid_exact_decimal",
            rows_decoded,
            f"SMAF-72 invalid exact decimal: {label}",
        ) from error


def _window(value: datetime) -> str | None:
    if WARMUP_START <= value < COMMON_START:
        return "warmup"
    if COMMON_START <= value < TRAIN_END:
        return "train"
    if TRAIN_END <= value < COMMON_END:
        return "selection"
    return None


def _project_rows(
    rows: Iterable[Mapping[str, Any]],
    allowlist: Sequence[str],
) -> Iterable[dict[str, str]]:
    required = set(allowlist)
    for index, row in enumerate(rows):
        if not required.issubset(row):
            raise SourceContractFailure(
                "schema_join_uniqueness_reconciliation",
                "missing_allowlist_column",
                index + 1,
                "SMAF-72 row misses an allowlisted column",
            )
        yield {column: str(row[column]) for column in allowlist}


def validate_operations(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[Operation, ...]:
    operations: list[Operation] = []
    seen: set[str] = set()
    for index, row in enumerate(_project_rows(rows, prereg.OPERATIONS_USECOLS)):
        operation_id = row["operation_id"]
        if not operation_id or "\x00" in operation_id:
            raise SourceContractFailure(
                "schema_join_uniqueness_reconciliation",
                "invalid_operation_id",
                index + 1,
                "SMAF-72 invalid operation_id",
            )
        if operation_id in seen:
            raise SourceContractFailure(
                "schema_join_uniqueness_reconciliation",
                "duplicate_operation_id",
                index + 1,
                "SMAF-72 duplicate operation_id",
            )
        seen.add(operation_id)
        operation_date_text = row["operation_date"]
        available_text = row["available_at_utc"]
        invalid_reasons: list[str] = []
        try:
            total_submitted = _exact_decimal(
                row["total_par_submitted"],
                f"operation[{index}] submitted",
                rows_decoded=index + 1,
            )
        except SourceContractFailure:
            total_submitted = None
            invalid_reasons.append("invalid_operation_total_submitted")
        try:
            total_accepted = _exact_decimal(
                row["total_par_accepted"],
                f"operation[{index}] accepted",
                rows_decoded=index + 1,
            )
        except SourceContractFailure:
            total_accepted = None
            invalid_reasons.append("invalid_operation_total_accepted")
        if (
            total_submitted is not None
            and total_accepted is not None
            and total_accepted > total_submitted
        ):
            invalid_reasons.append(
                "operation_total_accepted_above_submitted"
            )
        operation = Operation(
            operation_id=operation_id,
            operation_date=_parse_date(
                operation_date_text,
                f"operation[{index}] date",
                rows_decoded=index + 1,
            ),
            operation_date_text=operation_date_text,
            available_at=_parse_timestamp(
                available_text,
                f"operation[{index}] availability",
                rows_decoded=index + 1,
            ),
            available_at_text=available_text,
            total_submitted=total_submitted,
            total_accepted=total_accepted,
            invalid_reasons=tuple(invalid_reasons),
        )
        if not WARMUP_START <= operation.available_at < COMMON_END:
            raise SourceContractFailure(
                "schema_join_uniqueness_reconciliation",
                "operation_outside_frozen_window",
                index + 1,
                "SMAF-72 operation outside frozen window",
            )
        operations.append(operation)
    if not operations:
        raise SourceContractFailure(
            "schema_join_uniqueness_reconciliation",
            "empty_operations",
            0,
            "SMAF-72 operation source is empty",
        )
    return tuple(
        sorted(
            operations,
            key=lambda item: (item.available_at, item.operation_id),
        )
    )


def build_source(
    operation_rows: Iterable[Mapping[str, Any]],
    detail_rows: Iterable[Mapping[str, Any]],
) -> SourceBuild:
    operations = validate_operations(operation_rows)
    operation_map = {item.operation_id: item for item in operations}
    accumulators = {
        item.operation_id: Accumulator() for item in operations
    }
    unique_keys: set[tuple[str, str]] = set()
    invalid_reason_counts: Counter[str] = Counter()
    operation_reason_sets: dict[str, set[str]] = {
        item.operation_id: set(item.invalid_reasons)
        for item in operations
    }
    for operation in operations:
        invalid_reason_counts.update(operation.invalid_reasons)

    def invalidate(operation_id: str, reason: str) -> None:
        invalid_reason_counts[reason] += 1
        if reason not in operation_reason_sets[operation_id]:
            operation_reason_sets[operation_id].add(reason)

    detail_count = 0
    detail_rows_by_window: Counter[str] = Counter()
    valid_description_count = 0
    valid_description_rows_by_window: Counter[str] = Counter()
    for index, row in enumerate(_project_rows(detail_rows, prereg.DETAILS_USECOLS)):
        detail_count = index + 1
        operation_id = row["operation_id"]
        cusip = row["cusip"]
        if (
            not operation_id
            or not cusip
            or "\x00" in operation_id
            or "\x00" in cusip
        ):
            raise SourceContractFailure(
                "schema_join_uniqueness_reconciliation",
                "invalid_detail_identity",
                detail_count,
                "SMAF-72 invalid detail identity",
            )
        operation = operation_map.get(operation_id)
        if operation is None:
            raise SourceContractFailure(
                "schema_join_uniqueness_reconciliation",
                "unjoined_detail",
                detail_count,
                "SMAF-72 detail does not join an operation",
            )
        key = (operation_id, cusip)
        if key in unique_keys:
            raise SourceContractFailure(
                "schema_join_uniqueness_reconciliation",
                "duplicate_operation_cusip",
                detail_count,
                "SMAF-72 duplicate operation/CUSIP",
            )
        unique_keys.add(key)
        window = _window(operation.available_at)
        if window is None:
            raise SourceContractFailure(
                "schema_join_uniqueness_reconciliation",
                "detail_outside_frozen_window",
                detail_count,
                "SMAF-72 detail outside frozen window",
            )
        accumulator = accumulators[operation_id]
        accumulator.detail_rows += 1
        detail_rows_by_window[window] += 1
        if (
            row["operation_date"] != operation.operation_date_text
            or row["available_at_utc"] != operation.available_at_text
        ):
            invalidate(
                operation_id,
                "detail_join_value_mismatch",
            )

        try:
            submitted = _exact_decimal(
                row["par_submitted"],
                f"detail[{index}] submitted",
                rows_decoded=index + 1,
            )
        except SourceContractFailure:
            submitted = None
            invalidate(operation_id, "invalid_detail_par_submitted")
        try:
            accepted = _exact_decimal(
                row["par_accepted"],
                f"detail[{index}] accepted",
                rows_decoded=index + 1,
            )
        except SourceContractFailure:
            accepted = None
            invalidate(operation_id, "invalid_detail_par_accepted")
        try:
            available = _exact_decimal(
                row["actual_available_to_borrow"],
                f"detail[{index}] available",
                rows_decoded=index + 1,
            )
        except SourceContractFailure:
            available = None
            invalidate(
                operation_id,
                "invalid_detail_actual_available_to_borrow",
            )
        if (
            submitted is not None
            and accepted is not None
            and accepted > submitted
        ):
            invalidate(
                operation_id,
                "accepted_above_submitted",
            )
        try:
            tau = prereg.maturity_distance(
                operation.operation_date_text,
                row["security_description"],
            )
        except ValueError:
            tau = None
            invalidate(
                operation_id,
                "security_description_parser",
            )
        else:
            valid_description_count += 1
            valid_description_rows_by_window[window] += 1

        if submitted is not None:
            accumulator.submitted += submitted
            if tau is not None:
                accumulator.submitted_tau += submitted * tau
        if accepted is not None:
            accumulator.accepted += accepted
            if tau is not None:
                accumulator.accepted_tau += accepted * tau
        if available is not None:
            accumulator.available += available
            if tau is not None:
                accumulator.available_tau += available * tau

    features: list[OperationFeature] = []
    operations_by_window: Counter[str] = Counter()
    complete_operations_by_window: Counter[str] = Counter()
    for operation in operations:
        accumulator = accumulators[operation.operation_id]
        reasons = operation_reason_sets[operation.operation_id]
        if accumulator.detail_rows <= 0:
            invalidate(
                operation.operation_id,
                "operation_without_details",
            )
        else:
            submitted_reconcilable = not {
                "invalid_operation_total_submitted",
                "invalid_detail_par_submitted",
            }.intersection(reasons)
            accepted_reconcilable = not {
                "invalid_operation_total_accepted",
                "invalid_detail_par_accepted",
            }.intersection(reasons)
            if (
                (
                    submitted_reconcilable
                    and accumulator.submitted
                    != operation.total_submitted
                )
                or (
                    accepted_reconcilable
                    and accumulator.accepted
                    != operation.total_accepted
                )
            ):
                invalidate(
                    operation.operation_id,
                    "operation_total_reconciliation",
                )
            weights_parseable = not {
                "invalid_detail_par_submitted",
                "invalid_detail_par_accepted",
                "invalid_detail_actual_available_to_borrow",
            }.intersection(reasons)
            if weights_parseable and (
                accumulator.submitted <= 0
                or accumulator.accepted <= 0
                or accumulator.available <= 0
            ):
                invalidate(
                    operation.operation_id,
                    "nonpositive_centroid_weight",
                )

        window = _window(operation.available_at)
        if window is None:
            raise SourceContractFailure(
                "schema_join_uniqueness_reconciliation",
                "operation_outside_frozen_window",
                detail_count,
                "SMAF-72 operation outside frozen window",
            )
        operations_by_window[window] += 1
        valid = not operation_reason_sets[operation.operation_id]
        if valid:
            submitted_centroid = (
                accumulator.submitted_tau / accumulator.submitted
            )
            accepted_centroid = (
                accumulator.accepted_tau / accumulator.accepted
            )
            available_centroid = (
                accumulator.available_tau / accumulator.available
            )
            submitted_inventory = (
                submitted_centroid - available_centroid
            )
            submitted_award = submitted_centroid - accepted_centroid
            primary = submitted_inventory + submitted_award
            award_inventory = accepted_centroid - available_centroid
            demand_intensity = (
                accumulator.submitted / accumulator.available
            )
            complete_operations_by_window[window] += 1
        else:
            primary = None
            submitted_inventory = None
            submitted_award = None
            award_inventory = None
            demand_intensity = None
        features.append(
            OperationFeature(
                operation_id=operation.operation_id,
                operation_date=operation.operation_date,
                available_at=operation.available_at,
                available_at_text=operation.available_at_text,
                primary=primary,
                submitted_inventory_tilt=submitted_inventory,
                submitted_award_tilt=submitted_award,
                award_inventory_tilt=award_inventory,
                aggregate_demand_intensity=demand_intensity,
                detail_rows=accumulator.detail_rows,
                valid=valid,
                invalid_reasons=tuple(
                    sorted(operation_reason_sets[operation.operation_id])
                ),
            )
        )

    grouped: dict[datetime, list[OperationFeature]] = defaultdict(list)
    for feature in features:
        grouped[feature.available_at].append(feature)
    batches_by_window: Counter[str] = Counter()
    singleton_by_window: Counter[str] = Counter()
    for available_at, batch in grouped.items():
        window = _window(available_at)
        if window is None:
            raise RuntimeError("SMAF-72 batch outside frozen window")
        batches_by_window[window] += 1
        if len(batch) == 1 and batch[0].valid:
            singleton_by_window[window] += 1
    return SourceBuild(
        operations=operations,
        features=tuple(
            sorted(
                features,
                key=lambda item: (item.available_at, item.operation_id),
            )
        ),
        detail_rows=detail_count,
        valid_description_rows=valid_description_count,
        complete_operations=sum(
            feature.valid for feature in features
        ),
        batches=len(grouped),
        singleton_batches=sum(
            len(batch) == 1 and batch[0].valid
            for batch in grouped.values()
        ),
        invalid_reason_counts=dict(sorted(invalid_reason_counts.items())),
        detail_rows_by_window=dict(detail_rows_by_window),
        valid_description_rows_by_window=dict(
            valid_description_rows_by_window
        ),
        operations_by_window=dict(operations_by_window),
        complete_operations_by_window=dict(
            complete_operations_by_window
        ),
        batches_by_window=dict(batches_by_window),
        singleton_batches_by_window=dict(singleton_by_window),
    )


def _source_signal_id(
    control: str,
    feature: OperationFeature,
    tail: str,
) -> str:
    payload = (
        f"SMAF-72|{control}|{feature.operation_id}|"
        f"{feature.available_at_text}|{tail}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _side_control_id(control: str, parent_signal_id: str) -> str:
    payload = f"SMAF-72|{control}|{parent_signal_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _random_side(parent_signal_id: str) -> str:
    payload = f"SMAF-72|{parent_signal_id}|RANDOM_SIDE"
    first_byte = hashlib.sha256(payload.encode("utf-8")).digest()[0]
    return "LONG" if first_byte < 128 else "SHORT"


def _tail(value: Fraction, history: Sequence[Fraction]) -> str:
    if len(history) != prereg.Policy().history_operations:
        raise RuntimeError("SMAF-72 rank history length differs")
    lower = sum(prior < value for prior in history)
    equal = sum(prior == value for prior in history)
    numerator = 2 * lower + equal
    if 10 * numerator <= 252:
        return "LOW"
    if 10 * numerator >= 2268:
        return "HIGH"
    return "NEUTRAL"


def build_raw_signals(
    features: Sequence[OperationFeature],
) -> tuple[
    dict[str, list[Signal]],
    dict[str, dict[str, int]],
    tuple[OperationFeature, ...],
]:
    grouped: dict[datetime, list[OperationFeature]] = defaultdict(list)
    for feature in features:
        grouped[feature.available_at].append(feature)
    histories = {
        control: [] for control in prereg.SOURCE_CONTROL_ORDER
    }
    prior_states: dict[str, str | None] = {
        control: None for control in prereg.SOURCE_CONTROL_ORDER
    }
    raw = {control: [] for control in prereg.SOURCE_CONTROL_ORDER}
    audit = {
        split: {"rank_ready": 0, "LOW": 0, "HIGH": 0}
        for split in ("train", "selection")
    }
    segmented: list[OperationFeature] = []
    segment = 0
    history_length = prereg.Policy().history_operations
    for available_at in sorted(grouped):
        batch = sorted(
            grouped[available_at],
            key=lambda item: item.operation_id,
        )
        if len(batch) != 1 or not batch[0].valid:
            segment += 1
            for control in prereg.SOURCE_CONTROL_ORDER:
                histories[control].clear()
                prior_states[control] = None
            segmented.extend(replace(item, segment=-1) for item in batch)
            continue
        feature = replace(batch[0], segment=segment)
        segmented.append(feature)
        for control in prereg.SOURCE_CONTROL_ORDER:
            value = feature.value(control)
            history = histories[control]
            if len(history) < history_length:
                history.append(value)
                continue
            current_tail = _tail(value, history)
            split = _window(feature.available_at)
            if control == "primary" and split in audit:
                audit[split]["rank_ready"] += 1
                if current_tail in {"LOW", "HIGH"}:
                    audit[split][current_tail] += 1
            prior = prior_states[control]
            if (
                prior is not None
                and current_tail in {"LOW", "HIGH"}
                and current_tail != prior
            ):
                raw[control].append(
                    Signal(
                        control=control,
                        signal_id=_source_signal_id(
                            control,
                            feature,
                            current_tail,
                        ),
                        parent_signal_id="",
                        operation_id=feature.operation_id,
                        operation_date=feature.operation_date,
                        decision_time=feature.available_at,
                        segment=segment,
                        side=(
                            "LONG"
                            if current_tail == "LOW"
                            else "SHORT"
                        ),
                        tail=current_tail,
                    )
                )
            prior_states[control] = current_tail
            history.append(value)
            del history[:-history_length]
    return raw, audit, tuple(segmented)


def _ceil_to_5m(value: datetime) -> datetime:
    seconds = int(value.timestamp())
    aligned = ((seconds + BAR_SECONDS - 1) // BAR_SECONDS) * BAR_SECONDS
    return datetime.fromtimestamp(aligned + BAR_SECONDS, tz=UTC)


def _split_for(
    signal: Signal,
    entry_time: datetime,
    exit_time: datetime,
) -> str | None:
    operation_midnight = datetime.combine(
        signal.operation_date,
        time.min,
        tzinfo=UTC,
    )
    for name, start, end in (
        ("train", COMMON_START, TRAIN_END),
        ("selection", TRAIN_END, COMMON_END),
    ):
        if (
            start <= operation_midnight < end
            and start <= signal.decision_time < end
            and start <= entry_time < end
            and entry_time < exit_time <= end
        ):
            return name
    return None


def schedule_signals(
    control: str,
    signals: Sequence[Signal],
) -> tuple[list[Scheduled], list[Scheduled], Counter[str]]:
    candidates: list[Scheduled] = []
    for signal in signals:
        if signal.control != control:
            raise RuntimeError("SMAF-72 signal control mismatch")
        entry = signal.entry_override or _ceil_to_5m(signal.decision_time)
        exit_time = entry + timedelta(seconds=HOLD_SECONDS)
        candidates.append(
            Scheduled(
                control=control,
                signal_id=signal.signal_id,
                parent_signal_id=signal.parent_signal_id,
                operation_id=signal.operation_id,
                operation_date=signal.operation_date,
                decision_time=signal.decision_time,
                entry_time=entry,
                exit_time=exit_time,
                segment=signal.segment,
                side=signal.side,
                tail=signal.tail,
                split=None,
            )
        )
    candidates.sort(key=lambda item: (item.entry_time, item.signal_id))
    accepted_all: list[Scheduled] = []
    retained: list[Scheduled] = []
    reasons: Counter[str] = Counter()
    previous_exit: datetime | None = None
    for candidate in candidates:
        if previous_exit is not None and candidate.entry_time < previous_exit:
            reasons["overlap_suppressed"] += 1
            continue
        previous_exit = candidate.exit_time
        split = _split_for(
            Signal(
                control=candidate.control,
                signal_id=candidate.signal_id,
                parent_signal_id=candidate.parent_signal_id,
                operation_id=candidate.operation_id,
                operation_date=candidate.operation_date,
                decision_time=candidate.decision_time,
                segment=candidate.segment,
                side=candidate.side,
                tail=candidate.tail,
            ),
            candidate.entry_time,
            candidate.exit_time,
        )
        accepted = replace(candidate, split=split)
        accepted_all.append(accepted)
        if split is None:
            reasons["outside_or_crossing_split"] += 1
        else:
            retained.append(accepted)
    return accepted_all, retained, reasons


def _derive_primary_side_controls(
    primary_all: Sequence[Scheduled],
    primary_retained: Sequence[Scheduled],
    segmented_features: Sequence[OperationFeature],
) -> tuple[dict[str, list[Scheduled]], dict[str, dict[str, int]]]:
    controls: dict[str, list[Scheduled]] = {}
    diagnostics: dict[str, dict[str, int]] = {}
    for control in (
        "exact_direction_flip",
        "deterministic_random_side",
        "constant_long",
        "constant_short",
    ):
        rows: list[Scheduled] = []
        for parent in primary_retained:
            signal_id = _side_control_id(control, parent.signal_id)
            if control == "exact_direction_flip":
                side = "SHORT" if parent.side == "LONG" else "LONG"
            elif control == "deterministic_random_side":
                side = _random_side(parent.signal_id)
            elif control == "constant_long":
                side = "LONG"
            else:
                side = "SHORT"
            rows.append(
                replace(
                    parent,
                    control=control,
                    signal_id=signal_id,
                    parent_signal_id=parent.signal_id,
                    side=side,
                )
            )
        controls[control] = rows
        diagnostics[control] = {"retained": len(rows)}

    for control in ("one_extra_bar_delay", "one_operation_delay"):
        delayed: list[Signal] = []
        missing_successor = 0
        next_by_operation: dict[str, datetime] = {}
        if control == "one_operation_delay":
            by_segment: dict[int, list[OperationFeature]] = defaultdict(list)
            for feature in segmented_features:
                if feature.segment >= 0:
                    by_segment[feature.segment].append(feature)
            for segment_features in by_segment.values():
                ordered = sorted(
                    segment_features,
                    key=lambda item: (
                        item.available_at,
                        item.operation_id,
                    ),
                )
                for current, successor in pairwise(ordered):
                    next_by_operation[current.operation_id] = _ceil_to_5m(
                        successor.available_at
                    )
        for parent in primary_all:
            if control == "one_extra_bar_delay":
                entry = parent.entry_time + timedelta(seconds=BAR_SECONDS)
            else:
                entry = next_by_operation.get(parent.operation_id)
                if entry is None:
                    missing_successor += 1
                    continue
            delayed.append(
                Signal(
                    control=control,
                    signal_id=_side_control_id(
                        control,
                        parent.signal_id,
                    ),
                    parent_signal_id=parent.signal_id,
                    operation_id=parent.operation_id,
                    operation_date=parent.operation_date,
                    decision_time=parent.decision_time,
                    segment=parent.segment,
                    side=parent.side,
                    tail=parent.tail,
                    entry_override=entry,
                )
            )
        _, retained, reasons = schedule_signals(control, delayed)
        controls[control] = retained
        diagnostics[control] = {
            "raw_parents": len(primary_all),
            "missing_same_segment_successor": missing_successor,
            **dict(reasons),
            "retained": len(retained),
        }
    return controls, diagnostics


def build_clocks(
    features: Sequence[OperationFeature],
) -> tuple[
    dict[str, list[Scheduled]],
    dict[str, dict[str, int]],
    tuple[OperationFeature, ...],
    dict[str, Any],
]:
    raw, rank_audit, segmented = build_raw_signals(features)
    clocks: dict[str, list[Scheduled]] = {}
    schedule_diagnostics: dict[str, Any] = {}
    primary_all: list[Scheduled] = []
    primary_retained: list[Scheduled] = []
    for control in prereg.SOURCE_CONTROL_ORDER:
        accepted, retained, reasons = schedule_signals(
            control,
            raw[control],
        )
        clocks[control] = retained
        schedule_diagnostics[control] = {
            "raw": len(raw[control]),
            "accepted_global": len(accepted),
            **dict(reasons),
            "retained": len(retained),
        }
        if control == "primary":
            primary_all = accepted
            primary_retained = retained
    side_controls, side_diagnostics = _derive_primary_side_controls(
        primary_all,
        primary_retained,
        segmented,
    )
    clocks.update(side_controls)
    schedule_diagnostics.update(side_diagnostics)
    if tuple(clocks) != CONTROL_ORDER:
        raise RuntimeError("SMAF-72 control order differs")
    return clocks, rank_audit, segmented, schedule_diagnostics


def _fraction_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _share_at_least(
    numerator: int,
    denominator: int,
    minimum: Any,
) -> bool:
    if denominator <= 0:
        return False
    threshold = Fraction(str(minimum))
    return (
        numerator * threshold.denominator
        >= denominator * threshold.numerator
    )


def _share_at_most(
    numerator: int,
    denominator: int,
    maximum: Any,
) -> bool:
    if denominator <= 0:
        return False
    threshold = Fraction(str(maximum))
    return (
        numerator * threshold.denominator
        <= denominator * threshold.numerator
    )


def _coverage_report(
    source: SourceBuild,
) -> tuple[dict[str, Any], dict[str, bool], dict[str, bool]]:
    metrics: dict[str, Any] = {}
    parser_complete_checks: dict[str, bool] = {}
    singleton_checks: dict[str, bool] = {}
    for window in ("full", "warmup", "train", "selection"):
        if window == "full":
            details = source.detail_rows
            valid_descriptions = source.valid_description_rows
            operations = len(source.operations)
            complete_operations = source.complete_operations
            batches = source.batches
            singletons = source.singleton_batches
        else:
            details = source.detail_rows_by_window.get(window, 0)
            valid_descriptions = (
                source.valid_description_rows_by_window.get(window, 0)
            )
            operations = source.operations_by_window.get(window, 0)
            complete_operations = (
                source.complete_operations_by_window.get(window, 0)
            )
            batches = source.batches_by_window.get(window, 0)
            singletons = source.singleton_batches_by_window.get(window, 0)
        metrics[window] = {
            "description_rows_valid": valid_descriptions,
            "description_rows_total": details,
            "description_parser_coverage": _fraction_ratio(
                valid_descriptions,
                details,
            ),
            "complete_operations": complete_operations,
            "operation_rows": operations,
            "complete_operation_share": _fraction_ratio(
                complete_operations,
                operations,
            ),
            "singleton_batches": singletons,
            "availability_batches": batches,
            "single_operation_batch_share": _fraction_ratio(
                singletons,
                batches,
            ),
        }
        for key in (
            "description_parser_coverage",
            "complete_operation_share",
        ):
            parser_complete_checks[f"{window}:{key}"] = (
                metrics[window][key] == 1.0
            )
        singleton_checks[f"{window}:single_operation_batch_share"] = (
            metrics[window]["single_operation_batch_share"] == 1.0
        )
    return metrics, parser_complete_checks, singleton_checks


def _rank_report(
    audit: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    gates = prereg.build_manifest()["source_support_gates"]["coverage"]
    metrics: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for split, minimum in (
        ("train", gates["train_rank_ready_min"]),
        ("selection", gates["selection_rank_ready_min"]),
    ):
        rank_ready = int(audit[split]["rank_ready"])
        low = int(audit[split]["LOW"])
        high = int(audit[split]["HIGH"])
        split_metrics = {
            "rank_ready": rank_ready,
            "LOW": low,
            "HIGH": high,
            "LOW_share": _fraction_ratio(low, rank_ready),
            "HIGH_share": _fraction_ratio(high, rank_ready),
        }
        metrics[split] = split_metrics
        checks.update(
            _rank_checks_from_metrics(split_metrics, split, minimum)
        )
    return metrics, checks


def _rank_checks_from_metrics(
    metrics: Mapping[str, Any],
    split: str,
    minimum: int,
) -> dict[str, bool]:
    rank_ready = int(metrics["rank_ready"])
    low = int(metrics["LOW"])
    high = int(metrics["HIGH"])
    checks = {
        f"{split}:rank_ready_min": rank_ready >= minimum,
    }
    for tail_name, count in (("LOW", low), ("HIGH", high)):
        checks[f"{split}:{tail_name}_share_min"] = _share_at_least(
            count,
            rank_ready,
            Fraction(1, 20),
        )
        checks[f"{split}:{tail_name}_share_max"] = _share_at_most(
            count,
            rank_ready,
            Fraction(1, 5),
        )
    return checks


def _fixed_subperiods(split: str) -> dict[str, tuple[datetime, datetime]]:
    if split == "train":
        years = range(2020, 2023)
    elif split == "selection":
        years = range(2023, 2024)
    else:
        raise RuntimeError("SMAF-72 unknown split")
    periods: dict[str, tuple[datetime, datetime]] = {}
    for year in years:
        start = datetime(year, 1, 1, tzinfo=UTC)
        end = datetime(year + 1, 1, 1, tzinfo=UTC)
        periods[str(year)] = (start, end)
        periods[f"{year}H1"] = (
            start,
            datetime(year, 7, 1, tzinfo=UTC),
        )
        periods[f"{year}H2"] = (
            datetime(year, 7, 1, tzinfo=UTC),
            end,
        )
        for quarter in range(1, 5):
            month = 1 + (quarter - 1) * 3
            next_month = month + 3
            quarter_end = (
                datetime(year + 1, 1, 1, tzinfo=UTC)
                if next_month == 13
                else datetime(year, next_month, 1, tzinfo=UTC)
            )
            periods[f"{year}Q{quarter}"] = (
                datetime(year, month, 1, tzinfo=UTC),
                quarter_end,
            )
    return periods


def _event_stats(rows: Sequence[Scheduled], split: str) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda item: (item.entry_time, item.signal_id))
    side_counts = Counter(item.side for item in ordered)
    month_counts = Counter(
        item.entry_time.strftime("%Y-%m") for item in ordered
    )
    quarter_counts = Counter(
        f"{item.entry_time.year}Q{(item.entry_time.month - 1) // 3 + 1}"
        for item in ordered
    )
    subperiod_counts = {}
    for label, (start, end) in _fixed_subperiods(split).items():
        subperiod_counts[label] = sum(
            start <= item.entry_time and item.exit_time <= end
            for item in ordered
        )
    gaps = [
        (current.entry_time - prior.entry_time).total_seconds()
        for prior, current in pairwise(ordered)
    ]
    maximum_run = 0
    current_run = 0
    previous_side: str | None = None
    for item in ordered:
        if item.side == previous_side:
            current_run += 1
        else:
            current_run = 1
            previous_side = item.side
        maximum_run = max(maximum_run, current_run)
    total = len(ordered)
    return {
        "events": total,
        "LONG": side_counts["LONG"],
        "SHORT": side_counts["SHORT"],
        "LONG_share": _fraction_ratio(side_counts["LONG"], total),
        "SHORT_share": _fraction_ratio(side_counts["SHORT"], total),
        "active_months": len(month_counts),
        "month_counts": dict(sorted(month_counts.items())),
        "quarter_counts": dict(sorted(quarter_counts.items())),
        "subperiod_counts": subperiod_counts,
        "maximum_month_share": (
            max(month_counts.values()) / total
            if total and month_counts
            else None
        ),
        "maximum_quarter_share": (
            max(quarter_counts.values()) / total
            if total and quarter_counts
            else None
        ),
        "maximum_elapsed_entry_gap_days": (
            max(gaps) / 86_400 if gaps else None
        ),
        "maximum_same_side_run": maximum_run,
    }


def _event_checks(stats: Mapping[str, Any], split: str) -> dict[str, bool]:
    gate = prereg.build_manifest()["source_support_gates"][split]
    total = int(stats["events"])
    checks = {
        f"{split}:events_min": total >= gate["events_min"],
        f"{split}:events_max": total <= gate["events_max"],
        f"{split}:each_side_min": (
            stats["LONG"] >= gate["each_side_min"]
            and stats["SHORT"] >= gate["each_side_min"]
        ),
        f"{split}:each_side_share_min": (
            _share_at_least(
                int(stats["LONG"]),
                total,
                gate["each_side_share_min"],
            )
            and _share_at_least(
                int(stats["SHORT"]),
                total,
                gate["each_side_share_min"],
            )
        ),
        f"{split}:active_months_min": (
            stats["active_months"] >= gate["active_months_min"]
        ),
        f"{split}:maximum_month_share": (
            bool(stats["month_counts"])
            and _share_at_most(
                max(int(count) for count in stats["month_counts"].values()),
                total,
                gate["maximum_month_share"],
            )
        ),
        f"{split}:maximum_quarter_share": (
            bool(stats["quarter_counts"])
            and _share_at_most(
                max(
                    int(count)
                    for count in stats["quarter_counts"].values()
                ),
                total,
                gate["maximum_quarter_share"],
            )
        ),
        f"{split}:maximum_elapsed_entry_gap_days": (
            stats["maximum_elapsed_entry_gap_days"] is not None
            and stats["maximum_elapsed_entry_gap_days"]
            <= gate["maximum_elapsed_entry_gap_days"]
        ),
        f"{split}:maximum_same_side_run": (
            stats["maximum_same_side_run"]
            <= gate["maximum_same_side_run"]
        ),
    }
    periods = stats["subperiod_counts"]
    if split == "train":
        checks[f"{split}:each_year_min"] = all(
            periods[str(year)] >= gate["each_year_min"]
            for year in range(2020, 2023)
        )
        half_labels = [
            f"{year}H{half}"
            for year in range(2020, 2023)
            for half in (1, 2)
        ]
        quarter_labels = [
            f"{year}Q{quarter}"
            for year in range(2020, 2023)
            for quarter in range(1, 5)
        ]
    else:
        half_labels = ["2023H1", "2023H2"]
        quarter_labels = [f"2023Q{quarter}" for quarter in range(1, 5)]
    checks[f"{split}:each_half_min"] = all(
        periods[label] >= gate["each_half_min"] for label in half_labels
    )
    checks[f"{split}:each_quarter_min"] = all(
        periods[label] >= gate["each_quarter_min"]
        for label in quarter_labels
    )
    return checks


def _entry_jaccard(
    left: Sequence[Scheduled],
    right: Sequence[Scheduled],
) -> float | None:
    left_entries = {item.entry_time for item in left}
    right_entries = {item.entry_time for item in right}
    union = left_entries | right_entries
    if not union:
        return None
    return len(left_entries & right_entries) / len(union)


def _same_entry_side(
    primary: Sequence[Scheduled],
    control: Sequence[Scheduled],
) -> float | None:
    if not primary:
        return None
    control_map = {item.entry_time: item.side for item in control}
    matches = sum(
        control_map.get(item.entry_time) == item.side for item in primary
    )
    return matches / len(primary)


def _occupancy(
    rows: Sequence[Scheduled],
    start: datetime,
    end: datetime,
) -> npt.NDArray[np.int8]:
    size = int((end - start).total_seconds() // BAR_SECONDS)
    values = np.zeros(size, dtype=np.int8)
    for row in rows:
        if not (start <= row.entry_time < row.exit_time <= end):
            raise RuntimeError("SMAF-72 occupancy interval outside split")
        start_index = int(
            (row.entry_time - start).total_seconds() // BAR_SECONDS
        )
        end_index = int(
            (row.exit_time - start).total_seconds() // BAR_SECONDS
        )
        if (
            start + timedelta(seconds=start_index * BAR_SECONDS)
            != row.entry_time
            or start + timedelta(seconds=end_index * BAR_SECONDS)
            != row.exit_time
        ):
            raise RuntimeError("SMAF-72 occupancy interval off grid")
        if np.any(values[start_index:end_index]):
            raise RuntimeError("SMAF-72 occupancy self-overlap")
        values[start_index:end_index] = 1 if row.side == "LONG" else -1
    return values


def _occupancy_correlation(
    left: Sequence[Scheduled],
    right: Sequence[Scheduled],
    start: datetime,
    end: datetime,
) -> float | None:
    left_values = _occupancy(left, start, end)
    right_values = _occupancy(right, start, end)
    if (
        np.std(left_values) == 0
        or np.std(right_values) == 0
    ):
        return None
    value = float(np.corrcoef(left_values, right_values)[0, 1])
    return value if math.isfinite(value) else None


def _internal_checks_from_metrics(
    control: str,
    split: str,
    metrics: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, bool]:
    total = int(metrics["entries"])
    minimum = (
        gate["train_entries_min_each"]
        if split == "train"
        else gate["selection_entries_min_each"]
    )
    prefix = f"{control}:{split}"
    jaccard = metrics["exact_entry_jaccard"]
    reproduction = metrics["same_entry_same_side_reproduction"]
    correlation = metrics["signed_occupancy_pearson"]
    return {
        f"{prefix}:entries_min": total >= minimum,
        f"{prefix}:each_side_share_min": (
            _share_at_least(
                int(metrics["LONG"]),
                total,
                Fraction(1, 5),
            )
            and _share_at_least(
                int(metrics["SHORT"]),
                total,
                Fraction(1, 5),
            )
        ),
        f"{prefix}:entry_jaccard": (
            jaccard is not None
            and 0 <= jaccard <= gate["exact_entry_jaccard_max"]
        ),
        f"{prefix}:side_reproduction": (
            reproduction is not None
            and 0
            <= reproduction
            <= gate["same_entry_same_side_reproduction_max"]
        ),
        f"{prefix}:occupancy_correlation": (
            correlation is not None
            and -1 <= correlation <= 1
            and abs(correlation)
            <= gate["absolute_signed_occupancy_pearson_max"]
        ),
    }


def support_and_internal(
    source: SourceBuild,
    clocks: Mapping[str, Sequence[Scheduled]],
    rank_audit: Mapping[str, Mapping[str, int]],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, bool]],
]:
    (
        coverage_metrics,
        parser_complete_checks,
        singleton_checks,
    ) = _coverage_report(source)
    schema_checks = {
        reason: source.invalid_reason_counts.get(reason, 0) == 0
        for reason in SCHEMA_INVALID_REASON_CODES
    }
    parser_reason_checks = {
        reason: source.invalid_reason_counts.get(reason, 0) == 0
        for reason in PARSER_COMPLETE_INVALID_REASON_CODES
    }
    parser_reason_checks.update(parser_complete_checks)
    rank_metrics, rank_checks = _rank_report(rank_audit)
    primary_stats = {}
    event_checks: dict[str, bool] = {}
    for split in ("train", "selection"):
        rows = [
            item for item in clocks["primary"] if item.split == split
        ]
        primary_stats[split] = _event_stats(rows, split)
        event_checks.update(_event_checks(primary_stats[split], split))
    internal_metrics: dict[str, Any] = {}
    internal_checks: dict[str, bool] = {}
    gate = prereg.build_manifest()["source_support_gates"][
        "internal_component_distinctness"
    ]
    for control in prereg.SOURCE_CONTROL_ORDER[1:]:
        internal_metrics[control] = {}
        for split, start, end in (
            ("train", COMMON_START, TRAIN_END),
            ("selection", TRAIN_END, COMMON_END),
        ):
            primary_rows = [
                item for item in clocks["primary"] if item.split == split
            ]
            control_rows = [
                item for item in clocks[control] if item.split == split
            ]
            jaccard = _entry_jaccard(primary_rows, control_rows)
            reproduction = _same_entry_side(primary_rows, control_rows)
            correlation = _occupancy_correlation(
                primary_rows,
                control_rows,
                start,
                end,
            )
            side_counts = Counter(item.side for item in control_rows)
            total = len(control_rows)
            internal_metrics[control][split] = {
                "entries": total,
                "LONG": side_counts["LONG"],
                "SHORT": side_counts["SHORT"],
                "exact_entry_jaccard": jaccard,
                "same_entry_same_side_reproduction": reproduction,
                "signed_occupancy_pearson": correlation,
            }
            internal_checks.update(
                _internal_checks_from_metrics(
                    control,
                    split,
                    internal_metrics[control][split],
                    gate,
                )
            )
    return (
        {
            "coverage": coverage_metrics,
            "rank_selectivity": rank_metrics,
            "primary_event_support": primary_stats,
            "internal_component_distinctness": internal_metrics,
        },
        {
            "frozen_identity_and_exact_header": {"validated": True},
            "schema_join_uniqueness_reconciliation": schema_checks,
            "parser_coverage_and_complete_operations": (
                parser_reason_checks
            ),
            "singleton_causal_batches": singleton_checks,
            "rank_coverage_and_tail_selectivity": rank_checks,
            "primary_event_support": event_checks,
            "internal_component_distinctness": internal_checks,
        },
    )


def _all_true(checks: Mapping[str, bool]) -> bool:
    return bool(checks) and all(
        passed is True for passed in checks.values()
    )


def _expected_source_check_keys() -> dict[str, tuple[str, ...]]:
    windows = ("full", "warmup", "train", "selection")
    parser_complete = (
        *PARSER_COMPLETE_INVALID_REASON_CODES,
        *(
            f"{window}:{metric}"
            for window in windows
            for metric in (
                "description_parser_coverage",
                "complete_operation_share",
            )
        ),
    )
    singleton = tuple(
        f"{window}:single_operation_batch_share"
        for window in windows
    )
    rank = tuple(
        f"{split}:{suffix}"
        for split in ("train", "selection")
        for suffix in (
            "rank_ready_min",
            "LOW_share_min",
            "LOW_share_max",
            "HIGH_share_min",
            "HIGH_share_max",
        )
    )
    event: list[str] = []
    for split in ("train", "selection"):
        event.extend(
            f"{split}:{suffix}"
            for suffix in (
                "events_min",
                "events_max",
                "each_side_min",
                "each_side_share_min",
                "active_months_min",
                "maximum_month_share",
                "maximum_quarter_share",
                "maximum_elapsed_entry_gap_days",
                "maximum_same_side_run",
            )
        )
        if split == "train":
            event.append("train:each_year_min")
        event.extend(
            (
                f"{split}:each_half_min",
                f"{split}:each_quarter_min",
            )
        )
    internal = tuple(
        f"{control}:{split}:{suffix}"
        for control in prereg.SOURCE_CONTROL_ORDER[1:]
        for split in ("train", "selection")
        for suffix in (
            "entries_min",
            "each_side_share_min",
            "entry_jaccard",
            "side_reproduction",
            "occupancy_correlation",
        )
    )
    return {
        "frozen_identity_and_exact_header": ("validated",),
        "schema_join_uniqueness_reconciliation": (
            SCHEMA_INVALID_REASON_CODES
        ),
        "parser_coverage_and_complete_operations": parser_complete,
        "singleton_causal_batches": singleton,
        "rank_coverage_and_tail_selectivity": rank,
        "primary_event_support": tuple(event),
        "internal_component_distinctness": internal,
    }


def _expected_novelty_check_keys(
    preregistration: Mapping[str, Any],
) -> tuple[str, ...]:
    suffixes = (
        "minimum_rows",
        "exact_entry_jaccard",
        "same_entry_side",
        "candidate_24h",
        "comparator_24h",
        "occupancy_correlation",
    )
    return tuple(
        f"{contract['id']}:{group}:{suffix}"
        for contract in preregistration["novelty_contract"]["comparators"]
        for group in contract["selected_groups"]
        for suffix in suffixes
    )


def _validated_bool_checks(
    value: Any,
    expected: Sequence[str],
    label: str,
) -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise RuntimeError(f"SMAF-72 {label} check schema mismatch")
    checks: dict[str, bool] = {}
    for name in expected:
        passed = value[name]
        if type(passed) is not bool:
            raise RuntimeError(f"SMAF-72 {label} check is not boolean")
        checks[name] = passed
    return checks


def _required_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise RuntimeError(f"SMAF-72 {label} schema mismatch")
    return {str(key): item for key, item in value.items()}


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"SMAF-72 {label} is not a nonnegative integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise RuntimeError(f"SMAF-72 {label} is not numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise RuntimeError(f"SMAF-72 {label} is not finite")
    return normalized


def _validated_clock_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    clock = _required_mapping(payload.get("clock"), "clock evidence")
    if set(clock) != {
        "path",
        "sha256",
        "rows",
        "rows_by_control",
        "columns",
    }:
        raise RuntimeError("SMAF-72 clock evidence schema mismatch")
    if (
        not isinstance(clock["path"], str)
        or not isinstance(clock["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", clock["sha256"]) is None
        or clock["columns"] != list(CLOCK_COLUMNS)
    ):
        raise RuntimeError("SMAF-72 clock identity evidence invalid")
    rows_by_control = _required_mapping(
        clock["rows_by_control"],
        "clock control counts",
    )
    if set(rows_by_control) != set(CONTROL_ORDER):
        raise RuntimeError("SMAF-72 clock control schema mismatch")
    counts = {
        control: _nonnegative_int(
            rows_by_control[control],
            f"{control} clock rows",
        )
        for control in CONTROL_ORDER
    }
    total = _nonnegative_int(clock["rows"], "clock rows")
    if total != sum(counts.values()):
        raise RuntimeError("SMAF-72 clock row total mismatch")
    return counts


def _require_ratio(
    metrics: Mapping[str, Any],
    key: str,
    numerator: int,
    denominator: int,
    label: str,
) -> None:
    expected = _fraction_ratio(numerator, denominator)
    reported = metrics.get(key)
    if expected is None:
        matches = reported is None
    else:
        matches = (
            _finite_number(reported, f"{label} {key}") == expected
        )
    if not matches:
        raise RuntimeError(f"SMAF-72 {label} ratio evidence mismatch")


def _validated_event_metrics(
    value: Any,
    split: str,
) -> dict[str, Any]:
    metrics = _required_mapping(value, f"{split} event metrics")
    expected_fields = {
        "events",
        "LONG",
        "SHORT",
        "LONG_share",
        "SHORT_share",
        "active_months",
        "month_counts",
        "quarter_counts",
        "subperiod_counts",
        "maximum_month_share",
        "maximum_quarter_share",
        "maximum_elapsed_entry_gap_days",
        "maximum_same_side_run",
    }
    if set(metrics) != expected_fields:
        raise RuntimeError(f"SMAF-72 {split} event metric schema mismatch")
    total = _nonnegative_int(metrics["events"], f"{split} events")
    long_count = _nonnegative_int(metrics["LONG"], f"{split} LONG")
    short_count = _nonnegative_int(metrics["SHORT"], f"{split} SHORT")
    if long_count + short_count != total:
        raise RuntimeError(f"SMAF-72 {split} side count mismatch")
    _require_ratio(metrics, "LONG_share", long_count, total, split)
    _require_ratio(metrics, "SHORT_share", short_count, total, split)

    month_counts = _required_mapping(
        metrics["month_counts"],
        f"{split} month counts",
    )
    quarter_counts = _required_mapping(
        metrics["quarter_counts"],
        f"{split} quarter counts",
    )
    month_total = 0
    derived_quarters: Counter[str] = Counter()
    periods = _fixed_subperiods(split)
    for month_text, raw_count in month_counts.items():
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}", month_text) is None:
            raise RuntimeError(f"SMAF-72 {split} month key invalid")
        try:
            month_start = datetime.strptime(
                month_text,
                "%Y-%m",
            ).replace(tzinfo=UTC)
        except ValueError as error:
            raise RuntimeError(
                f"SMAF-72 {split} month key invalid"
            ) from error
        count = _nonnegative_int(
            raw_count,
            f"{split} month count",
        )
        if count <= 0:
            raise RuntimeError(f"SMAF-72 {split} empty active month")
        if not any(
            start <= month_start < end
            for start, end in periods.values()
        ):
            raise RuntimeError(f"SMAF-72 {split} month outside split")
        month_total += count
        quarter = (
            f"{month_start.year}Q"
            f"{(month_start.month - 1) // 3 + 1}"
        )
        derived_quarters[quarter] += count
    if month_total != total:
        raise RuntimeError(f"SMAF-72 {split} month count mismatch")
    if metrics["active_months"] != len(month_counts):
        raise RuntimeError(f"SMAF-72 {split} active month mismatch")

    validated_quarters = {
        quarter: _nonnegative_int(
            count,
            f"{split} quarter count",
        )
        for quarter, count in quarter_counts.items()
    }
    if validated_quarters != dict(sorted(derived_quarters.items())):
        raise RuntimeError(f"SMAF-72 {split} quarter count mismatch")
    subperiod_counts = _required_mapping(
        metrics["subperiod_counts"],
        f"{split} subperiod counts",
    )
    validated_periods = {
        label: _nonnegative_int(
            count,
            f"{split} subperiod count",
        )
        for label, count in subperiod_counts.items()
    }
    if set(validated_periods) != set(periods):
        raise RuntimeError(f"SMAF-72 {split} subperiod schema mismatch")
    if any(count > total for count in validated_periods.values()):
        raise RuntimeError(f"SMAF-72 {split} subperiod count invalid")
    years = range(2020, 2023) if split == "train" else range(2023, 2024)
    for year in years:
        year_count = validated_periods[str(year)]
        half_1 = validated_periods[f"{year}H1"]
        half_2 = validated_periods[f"{year}H2"]
        quarters = [
            validated_periods[f"{year}Q{quarter}"]
            for quarter in range(1, 5)
        ]
        year_month_entries = sum(
            count
            for month, count in month_counts.items()
            if month.startswith(f"{year}-")
        )
        if (
            year_count > year_month_entries
            or half_1 + half_2 > year_count
            or quarters[0] + quarters[1] > half_1
            or quarters[2] + quarters[3] > half_2
            or any(
                quarters[index]
                > validated_quarters.get(f"{year}Q{index + 1}", 0)
                for index in range(4)
            )
        ):
            raise RuntimeError(
                f"SMAF-72 {split} subperiod containment invalid"
            )
    if sum(validated_periods[str(year)] for year in years) > total:
        raise RuntimeError(f"SMAF-72 {split} yearly containment invalid")

    maximum_month = max(month_counts.values(), default=0)
    maximum_quarter = max(quarter_counts.values(), default=0)
    _require_ratio(
        metrics,
        "maximum_month_share",
        int(maximum_month),
        total,
        split,
    )
    _require_ratio(
        metrics,
        "maximum_quarter_share",
        int(maximum_quarter),
        total,
        split,
    )
    gap = metrics["maximum_elapsed_entry_gap_days"]
    if gap is not None and _finite_number(gap, f"{split} gap") < 0:
        raise RuntimeError(f"SMAF-72 {split} gap evidence invalid")
    maximum_run = _nonnegative_int(
        metrics["maximum_same_side_run"],
        f"{split} same-side run",
    )
    if (total == 0 and maximum_run != 0) or (
        total > 0 and not 1 <= maximum_run <= total
    ):
        raise RuntimeError(f"SMAF-72 {split} same-side run invalid")
    return metrics


def _recomputed_source_checks(
    payload: Mapping[str, Any],
    clock_counts: Mapping[str, int],
) -> dict[str, dict[str, bool]]:
    source = _required_mapping(payload.get("source"), "source evidence")
    support = _required_mapping(payload.get("support"), "support evidence")
    expected_source_fields = {
        "operation_rows",
        "detail_rows",
        "valid_description_rows",
        "operation_features",
        "complete_operations",
        "invalid_operations",
        "invalid_reason_counts",
        "availability_batches",
        "singleton_batches",
        "invalid_or_multi_operation_batches",
    }
    if set(source) != expected_source_fields:
        raise RuntimeError("SMAF-72 source evidence schema mismatch")
    if set(support) != {
        "coverage",
        "rank_selectivity",
        "primary_event_support",
        "internal_component_distinctness",
    }:
        raise RuntimeError("SMAF-72 support evidence schema mismatch")
    source_count_fields = (
        "operation_rows",
        "detail_rows",
        "valid_description_rows",
        "operation_features",
        "complete_operations",
        "invalid_operations",
        "availability_batches",
        "singleton_batches",
        "invalid_or_multi_operation_batches",
    )
    source_counts = {
        field: _nonnegative_int(source.get(field), f"source {field}")
        for field in source_count_fields
    }
    invalid_counts = _required_mapping(
        source.get("invalid_reason_counts"),
        "invalid reason counts",
    )
    allowed_reasons = {
        *SCHEMA_INVALID_REASON_CODES,
        *PARSER_COMPLETE_INVALID_REASON_CODES,
    }
    if not set(invalid_counts).issubset(allowed_reasons):
        raise RuntimeError("SMAF-72 invalid reason vocabulary mismatch")
    reason_counts = {
        reason: _nonnegative_int(count, f"{reason} count")
        for reason, count in invalid_counts.items()
    }
    schema_checks = {
        reason: reason_counts.get(reason, 0) == 0
        for reason in SCHEMA_INVALID_REASON_CODES
    }
    parser_checks = {
        reason: reason_counts.get(reason, 0) == 0
        for reason in PARSER_COMPLETE_INVALID_REASON_CODES
    }

    coverage = _required_mapping(
        support.get("coverage"),
        "coverage metrics",
    )
    windows = ("full", "warmup", "train", "selection")
    if set(coverage) != set(windows):
        raise RuntimeError("SMAF-72 coverage window schema mismatch")
    singleton_checks: dict[str, bool] = {}
    validated_coverage: dict[str, dict[str, Any]] = {}
    for window in windows:
        metrics = _required_mapping(
            coverage[window],
            f"{window} coverage metrics",
        )
        validated_coverage[window] = metrics
        expected_fields = {
            "description_rows_valid",
            "description_rows_total",
            "description_parser_coverage",
            "complete_operations",
            "operation_rows",
            "complete_operation_share",
            "singleton_batches",
            "availability_batches",
            "single_operation_batch_share",
        }
        if set(metrics) != expected_fields:
            raise RuntimeError(
                f"SMAF-72 {window} coverage metric schema mismatch"
            )
        valid_descriptions = _nonnegative_int(
            metrics["description_rows_valid"],
            f"{window} valid descriptions",
        )
        detail_rows = _nonnegative_int(
            metrics["description_rows_total"],
            f"{window} detail rows",
        )
        complete_operations = _nonnegative_int(
            metrics["complete_operations"],
            f"{window} complete operations",
        )
        operation_rows = _nonnegative_int(
            metrics["operation_rows"],
            f"{window} operation rows",
        )
        singleton_batches = _nonnegative_int(
            metrics["singleton_batches"],
            f"{window} singleton batches",
        )
        batches = _nonnegative_int(
            metrics["availability_batches"],
            f"{window} batches",
        )
        _require_ratio(
            metrics,
            "description_parser_coverage",
            valid_descriptions,
            detail_rows,
            window,
        )
        _require_ratio(
            metrics,
            "complete_operation_share",
            complete_operations,
            operation_rows,
            window,
        )
        _require_ratio(
            metrics,
            "single_operation_batch_share",
            singleton_batches,
            batches,
            window,
        )
        parser_checks[f"{window}:description_parser_coverage"] = (
            detail_rows > 0 and valid_descriptions == detail_rows
        )
        parser_checks[f"{window}:complete_operation_share"] = (
            operation_rows > 0
            and complete_operations == operation_rows
        )
        singleton_checks[
            f"{window}:single_operation_batch_share"
        ] = batches > 0 and singleton_batches == batches
    full = validated_coverage["full"]
    source_cross_checks = {
        "detail_rows": "description_rows_total",
        "valid_description_rows": "description_rows_valid",
        "complete_operations": "complete_operations",
        "operation_rows": "operation_rows",
        "availability_batches": "availability_batches",
        "singleton_batches": "singleton_batches",
    }
    for source_key, coverage_key in source_cross_checks.items():
        if source_counts[source_key] != full[coverage_key]:
            raise RuntimeError("SMAF-72 source/coverage count mismatch")
    for field in (
        "description_rows_valid",
        "description_rows_total",
        "complete_operations",
        "operation_rows",
        "singleton_batches",
        "availability_batches",
    ):
        if full[field] != sum(
            int(validated_coverage[window][field])
            for window in ("warmup", "train", "selection")
        ):
            raise RuntimeError("SMAF-72 coverage window count mismatch")
    if (
        source_counts["operation_features"]
        != source_counts["operation_rows"]
        or source_counts["invalid_operations"]
        != source_counts["operation_rows"]
        - source_counts["complete_operations"]
        or source_counts["invalid_or_multi_operation_batches"]
        != source_counts["availability_batches"]
        - source_counts["singleton_batches"]
    ):
        raise RuntimeError("SMAF-72 source structural count mismatch")

    rank_metrics = _required_mapping(
        support.get("rank_selectivity"),
        "rank metrics",
    )
    if set(rank_metrics) != {"train", "selection"}:
        raise RuntimeError("SMAF-72 rank metric schema mismatch")
    coverage_gates = prereg.build_manifest()["source_support_gates"][
        "coverage"
    ]
    rank_checks: dict[str, bool] = {}
    for split, minimum in (
        ("train", coverage_gates["train_rank_ready_min"]),
        ("selection", coverage_gates["selection_rank_ready_min"]),
    ):
        metrics = _required_mapping(
            rank_metrics[split],
            f"{split} rank metrics",
        )
        if set(metrics) != {
            "rank_ready",
            "LOW",
            "HIGH",
            "LOW_share",
            "HIGH_share",
        }:
            raise RuntimeError("SMAF-72 rank metric field mismatch")
        rank_ready = _nonnegative_int(
            metrics["rank_ready"],
            f"{split} rank ready",
        )
        low = _nonnegative_int(metrics["LOW"], f"{split} LOW")
        high = _nonnegative_int(metrics["HIGH"], f"{split} HIGH")
        complete_in_split = int(
            validated_coverage[split]["complete_operations"]
        )
        if low + high > rank_ready or rank_ready > complete_in_split:
            raise RuntimeError("SMAF-72 rank tail count mismatch")
        _require_ratio(metrics, "LOW_share", low, rank_ready, split)
        _require_ratio(metrics, "HIGH_share", high, rank_ready, split)
        rank_checks.update(
            _rank_checks_from_metrics(
                metrics,
                split,
                int(minimum),
            )
        )

    event_metrics = _required_mapping(
        support.get("primary_event_support"),
        "event metrics",
    )
    if set(event_metrics) != {"train", "selection"}:
        raise RuntimeError("SMAF-72 event metric schema mismatch")
    event_checks: dict[str, bool] = {}
    primary_events = 0
    for split in ("train", "selection"):
        metrics = _validated_event_metrics(event_metrics[split], split)
        primary_events += int(metrics["events"])
        event_checks.update(_event_checks(metrics, split))

    if clock_counts["primary"] != primary_events:
        raise RuntimeError("SMAF-72 primary clock/event count mismatch")

    internal_metrics = _required_mapping(
        support.get("internal_component_distinctness"),
        "internal metrics",
    )
    controls = tuple(prereg.SOURCE_CONTROL_ORDER[1:])
    if set(internal_metrics) != set(controls):
        raise RuntimeError("SMAF-72 internal control schema mismatch")
    internal_gate = prereg.build_manifest()["source_support_gates"][
        "internal_component_distinctness"
    ]
    internal_checks: dict[str, bool] = {}
    internal_entry_totals = {control: 0 for control in controls}
    for control in controls:
        splits = _required_mapping(
            internal_metrics[control],
            f"{control} internal metrics",
        )
        if set(splits) != {"train", "selection"}:
            raise RuntimeError("SMAF-72 internal split schema mismatch")
        for split in ("train", "selection"):
            metrics = _required_mapping(
                splits[split],
                f"{control} {split} internal metrics",
            )
            if set(metrics) != {
                "entries",
                "LONG",
                "SHORT",
                "exact_entry_jaccard",
                "same_entry_same_side_reproduction",
                "signed_occupancy_pearson",
            }:
                raise RuntimeError(
                    "SMAF-72 internal metric field mismatch"
                )
            entries = _nonnegative_int(
                metrics["entries"],
                f"{control} {split} entries",
            )
            long_count = _nonnegative_int(
                metrics["LONG"],
                f"{control} {split} LONG",
            )
            short_count = _nonnegative_int(
                metrics["SHORT"],
                f"{control} {split} SHORT",
            )
            if long_count + short_count != entries:
                raise RuntimeError("SMAF-72 internal side count mismatch")
            internal_entry_totals[control] += entries
            for field in (
                "exact_entry_jaccard",
                "same_entry_same_side_reproduction",
                "signed_occupancy_pearson",
            ):
                metric = metrics[field]
                normalized_metric = (
                    None
                    if metric is None
                    else _finite_number(metric, f"internal {field}")
                )
                if normalized_metric is not None and (
                    (
                        field != "signed_occupancy_pearson"
                        and not 0 <= normalized_metric <= 1
                    )
                    or (
                        field == "signed_occupancy_pearson"
                        and not -1 <= normalized_metric <= 1
                    )
                ):
                    raise RuntimeError(
                        "SMAF-72 internal metric outside range"
                    )
            internal_checks.update(
                _internal_checks_from_metrics(
                    control,
                    split,
                    metrics,
                    internal_gate,
                )
            )
    for control, entries in internal_entry_totals.items():
        if clock_counts[control] != entries:
            raise RuntimeError(
                "SMAF-72 internal clock/event count mismatch"
            )
    return {
        "frozen_identity_and_exact_header": {"validated": True},
        "schema_join_uniqueness_reconciliation": schema_checks,
        "parser_coverage_and_complete_operations": parser_checks,
        "singleton_causal_batches": singleton_checks,
        "rank_coverage_and_tail_selectivity": rank_checks,
        "primary_event_support": event_checks,
        "internal_component_distinctness": internal_checks,
    }


def first_source_failure(
    checks: Mapping[str, Mapping[str, bool]],
) -> tuple[str | None, str | None]:
    expected = _expected_source_check_keys()
    for stage in SOURCE_STAGE_ORDER:
        stage_checks = checks.get(stage, {})
        for name in expected[stage]:
            if stage_checks.get(name) is not True:
                return stage, name
    return None, None


def _parse_comparator_time(value: str, label: str) -> datetime:
    try:
        return _parse_timestamp(value, label)
    except SourceContractFailure as error:
        raise ComparatorContractFailure(
            "comparator_timestamp",
            0,
            str(error),
        ) from error


def read_comparator(
    contract: Mapping[str, Any],
) -> tuple[dict[str, list[Scheduled]], int]:
    path = Path(str(contract["path"]))
    opener = gzip.open if path.suffix == ".gz" else open
    groups: dict[str, list[Scheduled]] = defaultdict(list)
    entries_by_group: dict[str, set[datetime]] = defaultdict(set)
    previous_entry_by_group: dict[str, datetime] = {}
    previous_exit_by_group: dict[str, datetime] = {}
    rows_decoded = 0
    with opener(_path(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != prereg.csv_header(path):
            raise ComparatorContractFailure(
                "comparator_header",
                0,
                "SMAF-72 comparator header differs",
            )
        required = set(contract["read_csv"]["usecols"])
        if not required.issubset(reader.fieldnames or []):
            raise ComparatorContractFailure(
                "comparator_header",
                0,
                "SMAF-72 comparator allowlist missing",
            )
        allowed_groups = set(contract["allowed_groups"])
        side_map = contract["side_map"]
        for raw in reader:
            rows_decoded += 1
            group = str(raw["control"])
            if group not in allowed_groups:
                raise ComparatorContractFailure(
                    "comparator_group_vocabulary",
                    rows_decoded,
                    "SMAF-72 comparator has unknown group",
                )
            side_raw = str(raw["side"])
            if side_raw not in side_map:
                raise ComparatorContractFailure(
                    "comparator_side_vocabulary",
                    rows_decoded,
                    "SMAF-72 comparator has unknown side",
                )
            try:
                entry = _parse_comparator_time(
                    str(raw["entry_time"]),
                    f"{contract['id']} row {rows_decoded} entry",
                )
                exit_time = _parse_comparator_time(
                    str(raw["exit_time"]),
                    f"{contract['id']} row {rows_decoded} exit",
                )
            except ComparatorContractFailure as error:
                raise ComparatorContractFailure(
                    error.code,
                    rows_decoded,
                    str(error),
                ) from error
            if (
                entry >= exit_time
                or int(entry.timestamp()) % BAR_SECONDS
                or int(exit_time.timestamp()) % BAR_SECONDS
            ):
                raise ComparatorContractFailure(
                    "comparator_interval",
                    rows_decoded,
                    "SMAF-72 invalid comparator interval",
                )
            if entry in entries_by_group[group]:
                raise ComparatorContractFailure(
                    "comparator_duplicate_entry",
                    rows_decoded,
                    f"SMAF-72 comparator duplicate entry: {group}",
                )
            previous_entry = previous_entry_by_group.get(group)
            if previous_entry is not None and entry < previous_entry:
                raise ComparatorContractFailure(
                    "comparator_ordering",
                    rows_decoded,
                    f"SMAF-72 comparator ordering regression: {group}",
                )
            previous_exit = previous_exit_by_group.get(group)
            if previous_exit is not None and entry < previous_exit:
                raise ComparatorContractFailure(
                    "comparator_overlap",
                    rows_decoded,
                    f"SMAF-72 comparator overlap: {group}",
                )
            entries_by_group[group].add(entry)
            previous_entry_by_group[group] = entry
            previous_exit_by_group[group] = exit_time
            groups[group].append(
                Scheduled(
                    control=group,
                    signal_id=f"{contract['id']}:{rows_decoded}",
                    parent_signal_id="",
                    operation_id="",
                    operation_date=entry.date(),
                    decision_time=entry,
                    entry_time=entry,
                    exit_time=exit_time,
                    segment=0,
                    side=str(side_map[side_raw]),
                    tail="",
                    split=None,
                )
            )
    if set(groups) != set(contract["allowed_groups"]):
        raise ComparatorContractFailure(
            "comparator_group_vocabulary",
            rows_decoded,
            "SMAF-72 comparator full group vocabulary differs",
        )
    return dict(groups), rows_decoded


def _exact_entry_same_side(
    candidate: Sequence[Scheduled],
    comparator: Sequence[Scheduled],
) -> float | None:
    if not candidate:
        return None
    comparator_map = {item.entry_time: item.side for item in comparator}
    matches = sum(
        comparator_map.get(item.entry_time) == item.side
        for item in candidate
    )
    return matches / len(candidate)


def _within_24h_matches(
    candidate: Sequence[Scheduled],
    comparator: Sequence[Scheduled],
) -> int:
    left = sorted({item.entry_time for item in candidate})
    right = sorted({item.entry_time for item in comparator})
    left_index = 0
    right_index = 0
    matches = 0
    tolerance = timedelta(hours=24)
    while left_index < len(left) and right_index < len(right):
        left_time = left[left_index]
        right_time = right[right_index]
        if left_time < right_time - tolerance:
            left_index += 1
        elif right_time < left_time - tolerance:
            right_index += 1
        else:
            matches += 1
            left_index += 1
            right_index += 1
    return matches


def _novelty_checks_from_metrics(
    *,
    prefix: str,
    metrics: Mapping[str, Any],
    candidate_rows: int,
    minimum_rows: int,
    thresholds: Mapping[str, Any],
) -> dict[str, bool]:
    comparator_rows = int(metrics["contained_rows"])
    matches = int(metrics["within_24h_matches"])
    exact_jaccard = metrics["exact_entry_jaccard"]
    same_side = metrics["same_entry_same_side_reproduction"]
    correlation = metrics["signed_occupancy_pearson"]
    return {
        f"{prefix}:minimum_rows": comparator_rows >= minimum_rows,
        f"{prefix}:exact_entry_jaccard": (
            exact_jaccard is not None
            and 0
            <= exact_jaccard
            <= thresholds["exact_entry_jaccard_max"]
        ),
        f"{prefix}:same_entry_side": (
            same_side is not None
            and 0
            <= same_side
            <= thresholds["same_entry_same_side_reproduction_max"]
        ),
        f"{prefix}:candidate_24h": _share_at_most(
            matches,
            candidate_rows,
            thresholds["candidate_24h_containment_max"],
        ),
        f"{prefix}:comparator_24h": _share_at_most(
            matches,
            comparator_rows,
            thresholds["comparator_24h_containment_max"],
        ),
        f"{prefix}:occupancy_correlation": (
            correlation is not None
            and -1 <= correlation <= 1
            and abs(correlation)
            <= thresholds["absolute_signed_occupancy_pearson_max"]
        ),
    }


def evaluate_novelty(
    candidate: Sequence[Scheduled],
    preregistration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool], int]:
    contract = preregistration["novelty_contract"]
    thresholds = contract["thresholds_each_group"]
    report: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    rows_decoded = 0
    candidate_contained = [
        item
        for item in candidate
        if COMMON_START <= item.entry_time
        and item.exit_time <= COMMON_END
    ]
    if not candidate_contained:
        raise ComparatorContractFailure(
            "candidate_empty",
            0,
            "SMAF-72 novelty candidate is empty",
        )
    for comparator_contract in contract["comparators"]:
        try:
            groups, decoded = read_comparator(comparator_contract)
        except ComparatorContractFailure as error:
            raise ComparatorContractFailure(
                error.code,
                rows_decoded + error.rows_decoded,
                str(error),
            ) from error
        rows_decoded += decoded
        comparator_id = str(comparator_contract["id"])
        report[comparator_id] = {}
        selected_groups = set(comparator_contract["selected_groups"])
        for group in comparator_contract["allowed_groups"]:
            raw_rows = groups[group]
            rows = [
                item
                for item in raw_rows
                if COMMON_START <= item.entry_time
                and item.exit_time <= COMMON_END
            ]
            before = sum(
                item.exit_time <= COMMON_START for item in raw_rows
            )
            after = sum(item.entry_time >= COMMON_END for item in raw_rows)
            crossing = len(raw_rows) - len(rows) - before - after
            if crossing < 0:
                raise RuntimeError(
                    "SMAF-72 comparator containment accounting failed"
                )
            group_report: dict[str, Any] = {
                "selected_for_metrics": group in selected_groups,
                "raw_rows": len(raw_rows),
                "contained_rows": len(rows),
                "before_rows": before,
                "after_rows": after,
                "crossing_rows": crossing,
            }
            report[comparator_id][group] = group_report
            if group not in selected_groups:
                continue
            prefix = f"{comparator_id}:{group}"
            minimum = comparator_contract[
                "minimum_contained_rows_each_group"
            ]
            exact_jaccard = _entry_jaccard(candidate_contained, rows)
            same_side = _exact_entry_same_side(candidate_contained, rows)
            matches = _within_24h_matches(candidate_contained, rows)
            candidate_fraction = _fraction_ratio(
                matches,
                len(candidate_contained),
            )
            comparator_fraction = _fraction_ratio(matches, len(rows))
            correlation = _occupancy_correlation(
                candidate_contained,
                rows,
                COMMON_START,
                COMMON_END,
            )
            group_report.update(
                {
                    "exact_entry_jaccard": exact_jaccard,
                    "same_entry_same_side_reproduction": same_side,
                    "within_24h_matches": matches,
                    "candidate_24h_containment": candidate_fraction,
                    "comparator_24h_containment": comparator_fraction,
                    "signed_occupancy_pearson": correlation,
                }
            )
            checks.update(
                _novelty_checks_from_metrics(
                    prefix=prefix,
                    metrics=group_report,
                    candidate_rows=len(candidate_contained),
                    minimum_rows=int(minimum),
                    thresholds=thresholds,
                )
            )
    return report, checks, rows_decoded


def deterministic_clock_bytes(
    clocks: Mapping[str, Sequence[Scheduled]],
) -> bytes:
    text_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        text_buffer,
        fieldnames=list(CLOCK_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    for control in CONTROL_ORDER:
        for row in sorted(
            clocks[control],
            key=lambda item: (item.entry_time, item.signal_id),
        ):
            writer.writerow(
                {
                    "control": control,
                    "signal_id": row.signal_id,
                    "parent_signal_id": row.parent_signal_id,
                    "decision_time": canonical_time(row.decision_time),
                    "entry_time": canonical_time(row.entry_time),
                    "exit_time": canonical_time(row.exit_time),
                    "split": row.split,
                    "side": row.side,
                    "tail": row.tail,
                }
            )
    text = text_buffer.getvalue().encode("utf-8")
    header = text.splitlines()[0].decode("utf-8")
    if any(token in header.lower() for token in FORBIDDEN_CLOCK_TOKENS):
        raise RuntimeError("SMAF-72 clock header exposes forbidden values")
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=output,
        compresslevel=9,
        mtime=0,
    ) as zipped:
        zipped.write(text)
    return output.getvalue()


def _binding_report() -> dict[str, Any]:
    return {
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_document": {
            "path": str(PREREGISTRATION_DOCUMENT),
            "sha256": PREREGISTRATION_DOCUMENT_SHA256,
        },
        "preregistration_builder": {
            "path": str(PREREGISTRATION_BUILDER),
            "sha256": PREREGISTRATION_BUILDER_SHA256,
        },
        "preregistration_test": {
            "path": str(PREREGISTRATION_TEST),
            "sha256": PREREGISTRATION_TEST_SHA256,
        },
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "implementation_contract": {
            "path": str(IMPLEMENTATION_CONTRACT),
            "sha256": IMPLEMENTATION_CONTRACT_SHA256,
        },
        "active_frozen_dependencies": (
            prereg.active_frozen_dependencies()
        ),
        "implementation": {
            "script_path": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "test_path": str(TEST_PATH),
            "test_sha256": sha256_file(TEST_PATH),
        },
    }


def _clock_counts(
    clocks: Mapping[str, Sequence[Scheduled]],
) -> dict[str, int]:
    return {control: len(clocks[control]) for control in CONTROL_ORDER}


def build_report(
    *,
    source: SourceBuild,
    clocks: Mapping[str, Sequence[Scheduled]],
    rank_audit: Mapping[str, Mapping[str, int]],
    schedule_diagnostics: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    clock_bytes: bytes,
    clock_path: Path,
    protocol_git_subprocess_calls: int,
    comparator_loader: bool = True,
) -> dict[str, Any]:
    support, source_checks = support_and_internal(
        source,
        clocks,
        rank_audit,
    )
    first_stage, first_check = first_source_failure(source_checks)
    source_passed = first_stage is None
    novelty: dict[str, Any] = {}
    novelty_checks: dict[str, bool] = {}
    comparator_rows_decoded = 0
    comparator_failure: dict[str, Any] | None = None
    if source_passed and comparator_loader:
        try:
            (
                novelty,
                novelty_checks,
                comparator_rows_decoded,
            ) = evaluate_novelty(clocks["primary"], preregistration)
        except ComparatorContractFailure as error:
            comparator_rows_decoded = error.rows_decoded
            comparator_failure = {
                "code": error.code,
                "message": str(error),
            }
            novelty_checks = {
                f"comparator_contract:{error.code}": False,
            }
    novelty_passed = (
        source_passed
        and bool(novelty_checks)
        and all(novelty_checks.values())
        and comparator_failure is None
    )
    if source_passed and not novelty_passed:
        if comparator_failure is not None:
            first_check = next(iter(novelty_checks), None)
        else:
            for name in _expected_novelty_check_keys(preregistration):
                if novelty_checks.get(name) is not True:
                    first_check = name
                    break
        first_stage = "external_comparator_novelty"
        if first_check is None:
            first_check = "required_novelty_checks_missing"
    decision = (
        "advance_to_economic_evaluator_freeze"
        if novelty_passed
        else "retire_SMAF_72_unchanged_before_outcomes"
    )
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.POLICY_ID,
        "bindings": _binding_report(),
        "source": {
            "operation_rows": len(source.operations),
            "detail_rows": source.detail_rows,
            "valid_description_rows": source.valid_description_rows,
            "operation_features": len(source.features),
            "complete_operations": source.complete_operations,
            "invalid_operations": (
                len(source.operations) - source.complete_operations
            ),
            "invalid_reason_counts": dict(
                source.invalid_reason_counts
            ),
            "availability_batches": source.batches,
            "singleton_batches": source.singleton_batches,
            "invalid_or_multi_operation_batches": (
                source.batches - source.singleton_batches
            ),
        },
        "support": support,
        "source_checks": source_checks,
        "source_passed": source_passed,
        "schedule_diagnostics": dict(schedule_diagnostics),
        "novelty_authorized": source_passed,
        "comparator_rows_decoded": comparator_rows_decoded,
        "comparator_failure": comparator_failure,
        "novelty": novelty,
        "novelty_checks": novelty_checks,
        "novelty_passed": novelty_passed,
        "first_failing_stage": first_stage,
        "first_failing_check": first_check,
        "decision": decision,
        "clock": {
            "path": str(clock_path),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": sum(len(rows) for rows in clocks.values()),
            "rows_by_control": _clock_counts(clocks),
            "columns": list(CLOCK_COLUMNS),
        },
        "economic_evaluator_authorized": False,
        "outcomes_opened": False,
        "outcome_boundary": {
            "btc_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "forward_return_rows_loaded": 0,
            "pnl_cagr_mdd_opened": False,
            "network_calls": 0,
            "protocol_git_subprocess_calls": protocol_git_subprocess_calls,
            "model_or_gpu_calls": 0,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def build_source_failure_report(
    *,
    error: SourceContractFailure,
    clock_bytes: bytes,
    clock_path: Path,
    protocol_git_subprocess_calls: int,
) -> dict[str, Any]:
    clocks = {control: [] for control in CONTROL_ORDER}
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.POLICY_ID,
        "bindings": _binding_report(),
        "source": {
            "decoded_rows_before_failure": error.rows_decoded,
            "failure_code": error.code,
            "failure_message": str(error),
        },
        "support": {},
        "source_checks": {error.stage: {error.code: False}},
        "source_passed": False,
        "schedule_diagnostics": {},
        "novelty_authorized": False,
        "comparator_rows_decoded": 0,
        "comparator_failure": None,
        "novelty": {},
        "novelty_checks": {},
        "novelty_passed": False,
        "first_failing_stage": error.stage,
        "first_failing_check": error.code,
        "decision": "retire_SMAF_72_unchanged_before_outcomes",
        "clock": {
            "path": str(clock_path),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": 0,
            "rows_by_control": _clock_counts(clocks),
            "columns": list(CLOCK_COLUMNS),
        },
        "economic_evaluator_authorized": False,
        "outcomes_opened": False,
        "outcome_boundary": {
            "btc_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "forward_return_rows_loaded": 0,
            "pnl_cagr_mdd_opened": False,
            "network_calls": 0,
            "protocol_git_subprocess_calls": protocol_git_subprocess_calls,
            "model_or_gpu_calls": 0,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def build_support_from_rows(
    operation_rows: Iterable[Mapping[str, Any]],
    detail_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], bytes]:
    source = build_source(operation_rows, detail_rows)
    clocks, rank_audit, _, diagnostics = build_clocks(source.features)
    clock_bytes = deterministic_clock_bytes(clocks)
    report = build_report(
        source=source,
        clocks=clocks,
        rank_audit=rank_audit,
        schedule_diagnostics=diagnostics,
        preregistration=prereg.build_manifest(),
        clock_bytes=clock_bytes,
        clock_path=DEFAULT_CLOCK_OUTPUT,
        protocol_git_subprocess_calls=0,
        comparator_loader=False,
    )
    validate_report(report)
    return report, clock_bytes


def _validate_complete_novelty_evidence(
    value: Any,
    preregistration: Mapping[str, Any],
    rows_decoded: int,
    candidate_rows: int,
) -> dict[str, bool]:
    contracts = preregistration["novelty_contract"]["comparators"]
    comparator_ids = tuple(str(contract["id"]) for contract in contracts)
    if not isinstance(value, Mapping) or set(value) != set(comparator_ids):
        raise RuntimeError("SMAF-72 comparator evidence schema mismatch")
    count_fields = (
        "raw_rows",
        "contained_rows",
        "before_rows",
        "after_rows",
        "crossing_rows",
    )
    metric_fields = (
        "exact_entry_jaccard",
        "same_entry_same_side_reproduction",
        "within_24h_matches",
        "candidate_24h_containment",
        "comparator_24h_containment",
        "signed_occupancy_pearson",
    )
    raw_total = 0
    checks: dict[str, bool] = {}
    for contract in contracts:
        comparator_id = str(contract["id"])
        groups = value[comparator_id]
        allowed_groups = tuple(contract["allowed_groups"])
        selected_groups = set(contract["selected_groups"])
        if (
            not isinstance(groups, Mapping)
            or set(groups) != set(allowed_groups)
        ):
            raise RuntimeError("SMAF-72 comparator group schema mismatch")
        for group in allowed_groups:
            evidence = groups[group]
            selected = group in selected_groups
            expected_fields = {
                "selected_for_metrics",
                *count_fields,
                *(metric_fields if selected else ()),
            }
            if (
                not isinstance(evidence, Mapping)
                or set(evidence) != expected_fields
                or evidence.get("selected_for_metrics") is not selected
            ):
                raise RuntimeError(
                    "SMAF-72 comparator group evidence mismatch"
                )
            counts: dict[str, int] = {}
            for field in count_fields:
                count = evidence[field]
                if type(count) is not int or count < 0:
                    raise RuntimeError(
                        "SMAF-72 comparator count evidence invalid"
                    )
                counts[field] = count
            if counts["raw_rows"] <= 0 or counts["raw_rows"] != (
                counts["contained_rows"]
                + counts["before_rows"]
                + counts["after_rows"]
                + counts["crossing_rows"]
            ):
                raise RuntimeError(
                    "SMAF-72 comparator containment evidence invalid"
                )
            raw_total += counts["raw_rows"]
            if selected:
                matches = _nonnegative_int(
                    evidence["within_24h_matches"],
                    f"{comparator_id} {group} matches",
                )
                if (
                    matches > candidate_rows
                    or matches > counts["contained_rows"]
                ):
                    raise RuntimeError(
                        "SMAF-72 comparator match count invalid"
                    )
                _require_ratio(
                    evidence,
                    "candidate_24h_containment",
                    matches,
                    candidate_rows,
                    f"{comparator_id} {group}",
                )
                _require_ratio(
                    evidence,
                    "comparator_24h_containment",
                    matches,
                    counts["contained_rows"],
                    f"{comparator_id} {group}",
                )
                for field in (
                    "exact_entry_jaccard",
                    "same_entry_same_side_reproduction",
                    "signed_occupancy_pearson",
                ):
                    metric = evidence[field]
                    normalized_metric = (
                        None
                        if metric is None
                        else _finite_number(metric, f"comparator {field}")
                    )
                    if normalized_metric is not None and (
                        (
                            field != "signed_occupancy_pearson"
                            and not 0 <= normalized_metric <= 1
                        )
                        or (
                            field == "signed_occupancy_pearson"
                            and not -1 <= normalized_metric <= 1
                        )
                    ):
                        raise RuntimeError(
                            "SMAF-72 comparator metric outside range"
                        )
                prefix = f"{comparator_id}:{group}"
                checks.update(
                    _novelty_checks_from_metrics(
                        prefix=prefix,
                        metrics=evidence,
                        candidate_rows=candidate_rows,
                        minimum_rows=int(
                            contract[
                                "minimum_contained_rows_each_group"
                            ]
                        ),
                        thresholds=preregistration["novelty_contract"][
                            "thresholds_each_group"
                        ],
                    )
                )
    if rows_decoded <= 0 or raw_total != rows_decoded:
        raise RuntimeError("SMAF-72 comparator row evidence mismatch")
    return checks


def validate_report(payload: Mapping[str, Any]) -> None:
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("SMAF-72 support report hash mismatch")
    if payload.get("bindings") != _binding_report():
        raise RuntimeError("SMAF-72 support report binding mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("SMAF-72 support report opened outcomes")
    if payload.get("economic_evaluator_authorized") is not False:
        raise RuntimeError("SMAF-72 prematurely authorized economics")
    boundary = payload.get("outcome_boundary", {})
    for key in (
        "btc_market_rows_loaded",
        "funding_rows_loaded",
        "forward_return_rows_loaded",
        "network_calls",
        "model_or_gpu_calls",
    ):
        if boundary.get(key) != 0:
            raise RuntimeError(f"SMAF-72 forbidden boundary opened: {key}")
    if boundary.get("pnl_cagr_mdd_opened") is not False:
        raise RuntimeError("SMAF-72 PnL/CAGR/MDD opened")
    clock_counts = _validated_clock_counts(payload)
    rows_decoded = payload.get("comparator_rows_decoded")
    if type(rows_decoded) is not int or rows_decoded < 0:
        raise RuntimeError("SMAF-72 comparator row count invalid")

    expected_source_checks = _expected_source_check_keys()
    raw_source_checks = payload.get("source_checks")
    complete_source_report = (
        isinstance(raw_source_checks, Mapping)
        and set(raw_source_checks) == set(SOURCE_STAGE_ORDER)
    )
    source_failure: tuple[str, str] | None
    if complete_source_report:
        if not isinstance(raw_source_checks, Mapping):
            raise RuntimeError("SMAF-72 source check schema mismatch")
        source_checks = {
            stage: _validated_bool_checks(
                raw_source_checks[stage],
                expected_source_checks[stage],
                stage,
            )
            for stage in SOURCE_STAGE_ORDER
        }
        recomputed_source_checks = _recomputed_source_checks(
            payload,
            clock_counts,
        )
        if source_checks != recomputed_source_checks:
            raise RuntimeError(
                "SMAF-72 source metric/check contradiction"
            )
        source_checks_passed = all(
            _all_true(source_checks[stage])
            for stage in SOURCE_STAGE_ORDER
        )
        if source_checks_passed:
            source_failure = None
        else:
            failure_stage, failure_check = first_source_failure(
                source_checks
            )
            if failure_stage is None or failure_check is None:
                raise RuntimeError(
                    "SMAF-72 source failure evidence missing"
                )
            source_failure = (failure_stage, failure_check)
    else:
        if (
            not isinstance(raw_source_checks, Mapping)
            or len(raw_source_checks) != 1
        ):
            raise RuntimeError(
                "SMAF-72 terminal source check schema mismatch"
            )
        stage = next(iter(raw_source_checks))
        raw_stage_checks = raw_source_checks[stage]
        if (
            stage not in SOURCE_STAGE_ORDER
            or not isinstance(raw_stage_checks, Mapping)
            or len(raw_stage_checks) != 1
        ):
            raise RuntimeError(
                "SMAF-72 terminal source check schema mismatch"
            )
        name = next(iter(raw_stage_checks))
        if (
            not isinstance(name, str)
            or raw_stage_checks[name] is not False
        ):
            raise RuntimeError(
                "SMAF-72 terminal source check schema mismatch"
            )
        source_checks_passed = False
        source_failure = (str(stage), name)
        terminal_source = payload.get("source")
        if not isinstance(terminal_source, Mapping):
            raise RuntimeError(
                "SMAF-72 terminal source evidence mismatch"
            )
        decoded_before_failure = terminal_source.get(
            "decoded_rows_before_failure"
        )
        if (
            set(terminal_source)
            != {
                "decoded_rows_before_failure",
                "failure_code",
                "failure_message",
            }
            or type(decoded_before_failure) is not int
            or decoded_before_failure < 0
            or terminal_source.get("failure_code") != name
            or payload.get("support") != {}
            or payload.get("schedule_diagnostics") != {}
            or any(clock_counts.values())
        ):
            raise RuntimeError(
                "SMAF-72 terminal source evidence mismatch"
            )
    if payload.get("source_passed") is not source_checks_passed:
        raise RuntimeError("SMAF-72 source decision/check mismatch")
    if payload.get("novelty_authorized") is not source_checks_passed:
        raise RuntimeError("SMAF-72 novelty authorization mismatch")
    if not source_checks_passed:
        if (
            rows_decoded != 0
            or payload.get("comparator_failure") is not None
            or payload.get("novelty") != {}
            or payload.get("novelty_checks") != {}
        ):
            raise RuntimeError(
                "SMAF-72 comparator opened after source failure"
            )
        novelty_passed = False
        expected_first = source_failure
    else:
        comparator_failure = payload.get("comparator_failure")
        if comparator_failure is not None:
            if (
                not isinstance(comparator_failure, Mapping)
                or set(comparator_failure) != {"code", "message"}
                or not isinstance(comparator_failure.get("code"), str)
                or not comparator_failure["code"]
                or not isinstance(comparator_failure.get("message"), str)
                or payload.get("novelty") != {}
            ):
                raise RuntimeError(
                    "SMAF-72 comparator failure evidence mismatch"
                )
            failure_check = (
                f"comparator_contract:{comparator_failure['code']}"
            )
            novelty_checks = _validated_bool_checks(
                payload.get("novelty_checks"),
                (failure_check,),
                "terminal novelty",
            )
            if novelty_checks[failure_check] is not False:
                raise RuntimeError(
                    "SMAF-72 comparator failure check mismatch"
                )
            novelty_passed = False
            expected_first = (
                "external_comparator_novelty",
                failure_check,
            )
        else:
            preregistration = prereg.build_manifest()
            expected_novelty_checks = _expected_novelty_check_keys(
                preregistration
            )
            novelty_checks = _validated_bool_checks(
                payload.get("novelty_checks"),
                expected_novelty_checks,
                "novelty",
            )
            candidate_rows = clock_counts["primary"]
            recomputed_novelty_checks = (
                _validate_complete_novelty_evidence(
                payload.get("novelty"),
                preregistration,
                rows_decoded,
                    candidate_rows,
                )
            )
            if novelty_checks != recomputed_novelty_checks:
                raise RuntimeError(
                    "SMAF-72 novelty metric/check contradiction"
                )
            novelty_passed = _all_true(novelty_checks)
            failing_novelty_check = next(
                (
                    name
                    for name in expected_novelty_checks
                    if novelty_checks[name] is False
                ),
                None,
            )
            expected_first = (
                None
                if failing_novelty_check is None
                else (
                    "external_comparator_novelty",
                    failing_novelty_check,
                )
            )
    if payload.get("novelty_passed") is not novelty_passed:
        raise RuntimeError("SMAF-72 novelty decision/check mismatch")
    expected_decision = (
        "advance_to_economic_evaluator_freeze"
        if novelty_passed
        else "retire_SMAF_72_unchanged_before_outcomes"
    )
    if payload.get("decision") != expected_decision:
        raise RuntimeError("SMAF-72 support decision/check mismatch")
    expected_stage, expected_check = (
        expected_first if expected_first is not None else (None, None)
    )
    if (
        payload.get("first_failing_stage") != expected_stage
        or payload.get("first_failing_check") != expected_check
    ):
        raise RuntimeError("SMAF-72 first failure evidence mismatch")


def canonical_report_bytes(payload: Mapping[str, Any]) -> bytes:
    validate_report(payload)
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _git_check(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_protocol_committed() -> None:
    paths = (
        SCRIPT_PATH,
        TEST_PATH,
        IMPLEMENTATION_CONTRACT,
        PREREGISTRATION_DOCUMENT,
        PREREGISTRATION_BUILDER,
        PREREGISTRATION_TEST,
        PREREGISTRATION,
    )
    relative = [str(path) for path in paths]
    tracked = _git_check("ls-files", "--error-unmatch", "--", *relative)
    if tracked.returncode:
        raise RuntimeError("SMAF-72 protocol files are not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *relative)
    if clean.returncode:
        raise RuntimeError("SMAF-72 protocol files differ from HEAD")


def validate_preregistration() -> dict[str, Any]:
    expected = {
        PREREGISTRATION_DOCUMENT: PREREGISTRATION_DOCUMENT_SHA256,
        PREREGISTRATION_BUILDER: PREREGISTRATION_BUILDER_SHA256,
        PREREGISTRATION_TEST: PREREGISTRATION_TEST_SHA256,
        PREREGISTRATION: PREREGISTRATION_SHA256,
        IMPLEMENTATION_CONTRACT: IMPLEMENTATION_CONTRACT_SHA256,
    }
    for path, sha256 in expected.items():
        if sha256_file(path) != sha256:
            raise RuntimeError(f"SMAF-72 protocol binding drift: {path}")
    payload = json.loads(
        _path(PREREGISTRATION).read_text(encoding="utf-8")
    )
    prereg.validate_manifest(payload)
    if payload["manifest_hash"] != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("SMAF-72 preregistration manifest drift")
    prereg.validate_frozen_dependencies()
    return payload


def _read_csv_rows(
    path: Path,
    allowlist: Sequence[str],
) -> Iterable[dict[str, str]]:
    with gzip.open(
        _path(path),
        "rt",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        expected_header = prereg.csv_header(path)
        if reader.fieldnames != expected_header:
            raise SourceContractFailure(
                "frozen_identity_and_exact_header",
                "source_header",
                0,
                "SMAF-72 source header differs",
            )
        for row in reader:
            yield {column: str(row[column]) for column in allowlist}


def _assert_secure_io_capabilities() -> None:
    prereg._assert_secure_io_capabilities()


def _output_relative(path: str | Path) -> Path:
    candidate = Path(path)
    raw = str(path)
    if (
        candidate.is_absolute()
        or raw.startswith("~")
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("SMAF-72 output must be repository-relative")
    return candidate


def _open_parent(candidate: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current = os.open(REPOSITORY_ROOT, flags)
    try:
        for part in candidate.parent.parts:
            next_descriptor = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_descriptor
        return current
    except OSError as error:
        os.close(current)
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RuntimeError(
                "SMAF-72 output parent path is unsafe"
            ) from error
        raise


def _read_regular(directory: int, name: str) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as error:
        if error.errno in {
            errno.ELOOP,
            errno.ENOTDIR,
        }:
            raise RuntimeError("SMAF-72 output path is unsafe") from error
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("SMAF-72 output is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _publish_temporary(
    parent: int,
    temporary: str,
    output_name: str,
) -> None:
    os.link(
        temporary,
        output_name,
        src_dir_fd=parent,
        dst_dir_fd=parent,
        follow_symlinks=False,
    )


def _write_once(path: str | Path, payload: bytes) -> str:
    relative = _output_relative(path)
    _assert_secure_io_capabilities()
    parent = _open_parent(relative)
    temporary = (
        f".{relative.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    temporary_created = False
    try:
        try:
            existing = _read_regular(parent, relative.name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != payload:
                raise RuntimeError("SMAF-72 existing output is noncanonical")
            return "verified_existing"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        temporary_created = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        try:
            _publish_temporary(
                parent,
                temporary,
                relative.name,
            )
        except FileExistsError:
            if _read_regular(parent, relative.name) != payload:
                raise RuntimeError("SMAF-72 output publication race drift")
            return "verified_existing"
        os.fsync(parent)
        return "created"
    finally:
        if temporary_created:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.fsync(parent)
        os.close(parent)


def run(
    *,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
) -> tuple[dict[str, Any], dict[str, str]]:
    _assert_protocol_committed()
    preregistration = validate_preregistration()
    try:
        source = build_source(
            _read_csv_rows(prereg.OPERATIONS, prereg.OPERATIONS_USECOLS),
            _read_csv_rows(prereg.DETAILS, prereg.DETAILS_USECOLS),
        )
    except SourceContractFailure as error:
        clocks = {control: [] for control in CONTROL_ORDER}
        clock_bytes = deterministic_clock_bytes(clocks)
        report = build_source_failure_report(
            error=error,
            clock_bytes=clock_bytes,
            clock_path=Path(clock_output),
            protocol_git_subprocess_calls=2,
        )
    else:
        clocks, rank_audit, _, diagnostics = build_clocks(source.features)
        clock_bytes = deterministic_clock_bytes(clocks)
        report = build_report(
            source=source,
            clocks=clocks,
            rank_audit=rank_audit,
            schedule_diagnostics=diagnostics,
            preregistration=preregistration,
            clock_bytes=clock_bytes,
            clock_path=Path(clock_output),
            protocol_git_subprocess_calls=2,
            comparator_loader=True,
        )
    report_bytes = canonical_report_bytes(report)
    statuses = {
        "clock": _write_once(clock_output, clock_bytes),
        "report": _write_once(report_output, report_bytes),
    }
    return report, statuses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT))
    args = parser.parse_args()
    report, statuses = run(
        clock_output=args.clock_output,
        report_output=args.report_output,
    )
    print(
        json.dumps(
            {
                "statuses": statuses,
                "decision": report["decision"],
                "source_passed": report["source_passed"],
                "novelty_passed": report["novelty_passed"],
                "first_failing_stage": report["first_failing_stage"],
                "first_failing_check": report["first_failing_check"],
                "comparator_rows_decoded": report[
                    "comparator_rows_decoded"
                ],
                "outcomes_opened": report["outcomes_opened"],
                "manifest_hash": report["manifest_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
