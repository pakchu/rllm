"""Build frozen, outcome-blind CLOR-D1 source-language support evidence."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_collateral_liquidity_ordering_relation as prereg


UTC = timezone.utc
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = prereg.POLICY_ID
RUNNER_PATH = (
    "training/build_collateral_liquidity_ordering_relation_source_support.py"
)
TEST_PATH = (
    "tests/test_build_collateral_liquidity_ordering_relation_source_support.py"
)
CONTRACT_PATH = (
    "docs/clor-d1-source-support-implementation-contract-2026-07-25.md"
)
CONTRACT_COMMIT = "33525847b773fd76031d3cd31b848e780f5255f8"
CONTRACT_SHA256 = (
    "a77143311092f96d2c52065d5e32f07cbb78fb544fa9b5aa2456f4b97be5d490"
)
PREREGISTRATION_PATH = prereg.DEFAULT_OUTPUT
PREREGISTRATION_COMMIT = "41d77b9184c1f4bf839a7f44d963689afc44f7a5"
PREREGISTRATION_SHA256 = (
    "7aee03d42daade588a0e785133632ff6f9f9e2a8d23117b49ffd405a41341e89"
)
PREREGISTRATION_MANIFEST_HASH = (
    "881f4c631f924e26e827e71359ce8df1f9add309d0b828478a545c7262f00b2b"
)
PREREGISTRATION_CONTRACT_HASH = (
    "59b52c826eef8315dc81a15a0467a6a494c9fcad8c54f7bf2c73c8ecf344a22a"
)
PREREGISTRATION_PRODUCER_COMMIT = (
    "52efeed17445157ac585b246fde16830e23236b3"
)
PREREGISTRATION_PRODUCER_SHA256 = (
    "322b8d4e19649bcda4d99a5e2a7888f3e38908f443037e6c4569c7ab31075942"
)

EXECUTION_SEAL_PATH = (
    "results/clor_d1_source_support_execution_seal_2026-07-25.json"
)
SOURCE_OUTPUT = (
    "data/collateral_liquidity_ordering_relation_source_2020_2023.csv.gz"
)
CONTROL_OUTPUT = (
    "data/collateral_liquidity_ordering_relation_controls_2020_2023.csv.gz"
)
PASS_REPORT = (
    "results/collateral_liquidity_ordering_relation_source_support_2026-07-25.json"
)
REJECTION_REPORT = (
    "results/collateral_liquidity_ordering_relation_source_rejection_2026-07-25.json"
)
SEAL_PROTOCOL = "clor_d1_source_support_execution_seal_v1"
RESULT_PROTOCOL = "clor_d1_source_support_result_v1"
SELF_CHECK_PROTOCOL = "clor_d1_source_support_self_check_v1"
FAILURE_ACTION = "retire_clor_d1_unchanged_before_outcomes"
PASS_ACTION = "authorize_clor_d1_economic_rllm_evaluator_freeze_only"

SOURCE_ORDER = ("TREASURY", "SOMA", "OFR")
SOURCE_ORDER_INDEX = {name: index for index, name in enumerate(SOURCE_ORDER)}
FRESHNESS = {
    "TREASURY": timedelta(days=14),
    "SOMA": timedelta(days=4),
    "OFR": timedelta(days=4),
}
SPLITS = {
    "TRAIN": (
        datetime(2020, 9, 10, tzinfo=UTC),
        datetime(2022, 1, 1, tzinfo=UTC),
    ),
    "TEST": (
        datetime(2022, 1, 1, tzinfo=UTC),
        datetime(2023, 1, 1, tzinfo=UTC),
    ),
    "EVAL": (
        datetime(2023, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
    ),
}
GATE_NAMES = (
    "source_schema_chronology_reconciliation",
    "causal_schedule_split_append_invariance",
    "model_decision_count",
    "source_update_support",
    "maximum_decision_gap",
    "calendar_support",
    "primitive_diversity",
    "state_signature_concentration",
    "sequence_uniqueness",
    "relation_falsification_controls",
    "forbidden_access",
)
RELATION_CONTROLS = prereg.RELATION_FALSIFICATION_CONTROL_IDS
FORBIDDEN_COUNTER_NAMES = (
    "post_2023_source_value_rows_opened",
    "comparator_action_rows_opened",
    "market_rows_opened",
    "funding_rows_opened",
    "future_return_rows_built",
    "reward_rows_built",
    "model_rows_built",
    "selected_action_rows_built",
    "trade_rows_built",
    "pnl_cagr_mdd_values_computed",
    "network_calls",
)
SOURCE_COLUMNS = (
    "split",
    "execution_time",
    "valid",
    "invalid_reason",
    "model_decision",
    "updated",
    "treasury",
    "soma_submitted_step",
    "soma_accepted_step",
    "soma_coverage_step",
    "ofr_rate_order",
    "ofr_volume_order",
    "line_text",
    "line_sha256",
    "sequence_sha256",
    "decision_expiry_time",
)
CONTROL_COLUMNS = ("control", *SOURCE_COLUMNS)
HEX40 = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
PLAIN_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z", re.ASCII)
CANONICAL_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z", re.ASCII)


@dataclass(frozen=True)
class SourceBatch:
    source: str
    available_at: datetime
    valid: bool
    token: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceState:
    available_at: datetime
    valid: bool
    token: tuple[str, ...] = ()


@dataclass(frozen=True)
class JointRow:
    split: str
    execution_time: datetime
    valid: bool
    invalid_reason: str
    model_decision: bool
    updated: tuple[str, ...]
    treasury: str
    soma_submitted_step: str
    soma_accepted_step: str
    soma_coverage_step: str
    ofr_rate_order: str
    ofr_volume_order: str
    line_text: str
    line_sha256: str
    sequence_sha256: str
    decision_expiry_time: datetime | None

    def csv_row(self) -> dict[str, str]:
        return {
            "split": self.split,
            "execution_time": format_time(self.execution_time),
            "valid": "1" if self.valid else "0",
            "invalid_reason": self.invalid_reason,
            "model_decision": "1" if self.model_decision else "0",
            "updated": "|".join(self.updated),
            "treasury": self.treasury,
            "soma_submitted_step": self.soma_submitted_step,
            "soma_accepted_step": self.soma_accepted_step,
            "soma_coverage_step": self.soma_coverage_step,
            "ofr_rate_order": self.ofr_rate_order,
            "ofr_volume_order": self.ofr_volume_order,
            "line_text": self.line_text,
            "line_sha256": self.line_sha256,
            "sequence_sha256": self.sequence_sha256,
            "decision_expiry_time": (
                ""
                if self.decision_expiry_time is None
                else format_time(self.decision_expiry_time)
            ),
        }


@dataclass
class AccessLedger:
    treasury_rows: int = 0
    soma_operation_rows: int = 0
    soma_detail_rows: int = 0
    ofr_rows: int = 0
    predecessor_value_rows_opened: int = 0

    def decoded_rows(self) -> dict[str, int]:
        return {
            "treasury": self.treasury_rows,
            "soma_operations": self.soma_operation_rows,
            "soma_details": self.soma_detail_rows,
            "ofr": self.ofr_rows,
            "predecessors": self.predecessor_value_rows_opened,
        }


def repository_path(path: str | Path) -> Path:
    return prereg.repository_path(path)


def sha256_file(path: str | Path) -> str:
    return prereg.sha256_file(path)


def canonical_bytes(payload: Any) -> bytes:
    return prereg.canonical_bytes(payload)


def canonical_hash(payload: Any) -> str:
    return prereg.canonical_hash(payload)


def json_bytes(payload: Mapping[str, Any]) -> bytes:
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


def forbidden_access() -> dict[str, int]:
    return {name: 0 for name in FORBIDDEN_COUNTER_NAMES}


def parse_date(value: Any) -> date:
    if not isinstance(value, str) or not CANONICAL_DATE.fullmatch(value):
        raise RuntimeError("CLOR-D1 date is not canonical")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise RuntimeError("CLOR-D1 date is invalid") from error
    if parsed.isoformat() != value:
        raise RuntimeError("CLOR-D1 date round-trip changed")
    return parsed


def parse_source_time(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise RuntimeError("CLOR-D1 source timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RuntimeError("CLOR-D1 source timestamp is invalid") from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timedelta(0)
        or parsed.microsecond
        or parsed.isoformat() != value
    ):
        raise RuntimeError("CLOR-D1 source timestamp is not canonical UTC")
    parsed = parsed.astimezone(UTC)
    if parsed >= datetime(2024, 1, 1, tzinfo=UTC):
        raise RuntimeError("CLOR-D1 source timestamp reached sealed period")
    return parsed


def parse_canonical_time(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError("CLOR-D1 canonical timestamp must end in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise RuntimeError("CLOR-D1 canonical timestamp is invalid") from error
    if parsed.microsecond or format_time(parsed) != value:
        raise RuntimeError("CLOR-D1 canonical timestamp round-trip changed")
    return parsed


def format_time(value: datetime) -> str:
    if (
        value.tzinfo is None
        or value.utcoffset() != timedelta(0)
        or value.microsecond
    ):
        raise RuntimeError("CLOR-D1 output timestamp is not whole-second UTC")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_fraction(
    value: Any,
    *,
    allow_empty: bool = False,
    nonnegative: bool = False,
) -> Fraction | None:
    if allow_empty and value == "":
        return None
    if not isinstance(value, str) or not PLAIN_DECIMAL.fullmatch(value):
        raise RuntimeError("CLOR-D1 decimal is not canonical")
    try:
        decimal = Decimal(value)
    except InvalidOperation as error:
        raise RuntimeError("CLOR-D1 decimal is invalid") from error
    if not decimal.is_finite():
        raise RuntimeError("CLOR-D1 decimal is non-finite")
    result = Fraction(decimal)
    if result == 0 and value.startswith("-"):
        raise RuntimeError("CLOR-D1 decimal uses signed zero")
    if nonnegative and result < 0:
        raise RuntimeError("CLOR-D1 decimal is not canonical nonnegative")
    return result


def weak_order(
    values: Mapping[str, Fraction],
    label_order: Sequence[str],
) -> str:
    if set(values) != set(label_order):
        raise RuntimeError("CLOR-D1 weak-order labels differ")
    levels = sorted(set(values.values()), reverse=True)
    groups = [
        "=".join(label for label in label_order if values[label] == level)
        for level in levels
    ]
    return ">".join(groups)


def execution_time(available_at: datetime) -> datetime:
    if available_at.tzinfo is None or available_at.utcoffset() != timedelta(0):
        raise RuntimeError("CLOR-D1 availability is not UTC")
    if available_at.microsecond:
        raise RuntimeError("CLOR-D1 availability has fractional seconds")
    seconds = available_at.minute * 60 + available_at.second
    remainder = seconds % 300
    rounded = available_at
    if remainder:
        rounded += timedelta(seconds=300 - remainder)
    return rounded + timedelta(minutes=5)


def split_for(value: datetime) -> str | None:
    for split, (start, end) in SPLITS.items():
        if start <= value < end:
            return split
    return None


def _git_output(*args: str) -> str:
    return prereg._git_output(*args)


def _assert_committed(path: str, expected_commit: str | None = None) -> str:
    return prereg._assert_committed(path, expected_commit=expected_commit)


def _git_blob_sha256(commit: str, path: str) -> str:
    return prereg._git_blob_sha256(commit, path)


def _worktree_clean() -> bool:
    return not _git_output("status", "--porcelain", "--untracked-files=all")


def _binding(path: str, commit: str, digest: str) -> dict[str, str]:
    return {"path": path, "commit": commit, "sha256": digest}


def _validate_binding(path: str, commit: str, digest: str) -> None:
    if (
        _assert_committed(path, expected_commit=commit) != commit
        or sha256_file(path) != digest
        or _git_blob_sha256(commit, path) != digest
    ):
        raise RuntimeError(f"CLOR-D1 frozen authority mismatch: {path}")


def python_runtime() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    pandas_file = Path(pd.__file__).resolve()
    return {
        "python": {
            "path": str(executable),
            "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "version": sys.version,
        },
        "pandas": {
            "path": str(pandas_file),
            "sha256": hashlib.sha256(pandas_file.read_bytes()).hexdigest(),
            "version": pd.__version__,
        },
        "git": prereg.validate_runtime_authority(),
    }


def static_authority() -> dict[str, Any]:
    _validate_binding(CONTRACT_PATH, CONTRACT_COMMIT, CONTRACT_SHA256)
    _validate_binding(
        prereg.BOUNDARY_DOCUMENT,
        prereg.BOUNDARY_COMMIT,
        prereg.BOUNDARY_SHA256,
    )
    _validate_binding(
        prereg.PRODUCER_SCRIPT,
        PREREGISTRATION_PRODUCER_COMMIT,
        PREREGISTRATION_PRODUCER_SHA256,
    )
    _validate_binding(
        PREREGISTRATION_PATH,
        PREREGISTRATION_COMMIT,
        PREREGISTRATION_SHA256,
    )
    payload = json.loads(repository_path(PREREGISTRATION_PATH).read_text())
    prereg.validate_manifest(payload)
    if (
        payload["manifest_hash"] != PREREGISTRATION_MANIFEST_HASH
        or payload["scientific_contract_hash"]
        != PREREGISTRATION_CONTRACT_HASH
        or payload["authority"]["producer"]
        != _binding(
            prereg.PRODUCER_SCRIPT,
            PREREGISTRATION_PRODUCER_COMMIT,
            PREREGISTRATION_PRODUCER_SHA256,
        )
        or payload["source_values_opened"] is not False
        or payload["outcomes_opened"] is not False
    ):
        raise RuntimeError("CLOR-D1 preregistration authority mismatch")
    source_authority = prereg.validate_frozen_authority()
    return {
        "runtime": python_runtime(),
        "contract": _binding(
            CONTRACT_PATH,
            CONTRACT_COMMIT,
            CONTRACT_SHA256,
        ),
        "boundary": _binding(
            prereg.BOUNDARY_DOCUMENT,
            prereg.BOUNDARY_COMMIT,
            prereg.BOUNDARY_SHA256,
        ),
        "preregistration": {
            **_binding(
                PREREGISTRATION_PATH,
                PREREGISTRATION_COMMIT,
                PREREGISTRATION_SHA256,
            ),
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
            "scientific_contract_hash": PREREGISTRATION_CONTRACT_HASH,
        },
        "preregistration_producer": _binding(
            prereg.PRODUCER_SCRIPT,
            PREREGISTRATION_PRODUCER_COMMIT,
            PREREGISTRATION_PRODUCER_SHA256,
        ),
        "source_authority_hash": canonical_hash(source_authority),
    }


def _projected_csv(
    path: str,
    usecols: Sequence[str],
    *,
    ledger: AccessLedger,
    ledger_field: str,
) -> pd.DataFrame:
    frame = pd.read_csv(
        repository_path(path),
        compression="gzip",
        usecols=list(usecols),
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    if tuple(frame.columns) != tuple(usecols):
        raise RuntimeError(f"CLOR-D1 projected columns changed: {path}")
    setattr(ledger, ledger_field, len(frame))
    return frame


def load_source_frames(
    ledger: AccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    treasury = _projected_csv(
        prereg.TREASURY_PATH,
        prereg.TREASURY_ALLOWLIST,
        ledger=ledger,
        ledger_field="treasury_rows",
    )
    soma_operations = _projected_csv(
        prereg.SOMA_OPERATIONS_PATH,
        prereg.SOMA_OPERATION_ALLOWLIST,
        ledger=ledger,
        ledger_field="soma_operation_rows",
    )
    soma_details = _projected_csv(
        prereg.SOMA_DETAILS_PATH,
        prereg.SOMA_DETAIL_ALLOWLIST,
        ledger=ledger,
        ledger_field="soma_detail_rows",
    )
    ofr = _projected_csv(
        prereg.OFR_PATH,
        prereg.OFR_ALLOWLIST,
        ledger=ledger,
        ledger_field="ofr_rows",
    )
    return treasury, soma_operations, soma_details, ofr


def build_treasury_batches(
    frame: pd.DataFrame,
    *,
    enforce_physical_counts: bool = False,
) -> tuple[list[SourceBatch], dict[str, Any]]:
    if tuple(frame.columns) != prereg.TREASURY_ALLOWLIST:
        raise RuntimeError("CLOR-D1 Treasury allowlist changed")
    term_index = {
        term: index for index, term in enumerate(prereg.TREASURY_TERM_ORDER)
    }
    rows_by_time: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[date, datetime, str]] = set()
    previous_date: date | None = None
    complete_count = incomplete_count = 0
    for raw in frame.to_dict(orient="records"):
        auction_day = parse_date(raw["auction_date"])
        available = parse_source_time(raw["result_available_at_utc"])
        if not date(2016, 1, 1) <= auction_day <= date(2023, 12, 31):
            raise RuntimeError("CLOR-D1 Treasury date left frozen window")
        if previous_date is not None and auction_day < previous_date:
            raise RuntimeError("CLOR-D1 Treasury dates are not nondecreasing")
        previous_date = auction_day
        term = raw["original_security_term"]
        if term not in term_index:
            raise RuntimeError("CLOR-D1 Treasury term changed")
        identity = (auction_day, available, term)
        if identity in identities:
            raise RuntimeError("CLOR-D1 Treasury physical identity duplicated")
        identities.add(identity)
        complete_text = raw["source_complete"]
        if complete_text not in {"true", "false"}:
            raise RuntimeError("CLOR-D1 Treasury completeness changed")
        complete = complete_text == "true"
        amount_fields = (
            "competitive_accepted_usd",
            "primary_dealer_accepted_usd",
            "direct_bidder_accepted_usd",
            "indirect_bidder_accepted_usd",
        )
        if not complete:
            incomplete_count += 1
            if any(raw[field] != "" for field in amount_fields):
                raise RuntimeError(
                    "CLOR-D1 incomplete Treasury row has an amount"
                )
            rows_by_time[available].append(
                {"complete": False, "term": term}
            )
            continue
        complete_count += 1
        competitive = parse_fraction(
            raw["competitive_accepted_usd"],
            nonnegative=True,
        )
        primary = parse_fraction(
            raw["primary_dealer_accepted_usd"],
            nonnegative=True,
        )
        direct = parse_fraction(
            raw["direct_bidder_accepted_usd"],
            nonnegative=True,
        )
        indirect = parse_fraction(
            raw["indirect_bidder_accepted_usd"],
            nonnegative=True,
        )
        if (
            competitive is None
            or primary is None
            or direct is None
            or indirect is None
            or competitive <= 0
            or primary + direct + indirect != competitive
        ):
            raise RuntimeError("CLOR-D1 Treasury amounts do not reconcile")
        order = weak_order(
            {"P": primary, "D": direct, "I": indirect},
            ("P", "D", "I"),
        )
        rows_by_time[available].append(
            {
                "complete": True,
                "term": term,
                "order": order,
            }
        )
    if enforce_physical_counts and (
        len(frame) != 445 or complete_count != 440 or incomplete_count != 5
    ):
        raise RuntimeError("CLOR-D1 Treasury physical counts changed")
    batches: list[SourceBatch] = []
    for available in sorted(rows_by_time):
        rows = rows_by_time[available]
        if any(not row["complete"] for row in rows):
            batches.append(
                SourceBatch("TREASURY", available, False)
            )
            continue
        terms = [row["term"] for row in rows]
        if len(terms) != len(set(terms)):
            raise RuntimeError("CLOR-D1 Treasury batch term duplicated")
        ordered = sorted(rows, key=lambda row: term_index[row["term"]])
        token = "|".join(
            f"{row['term']}:{row['order']}" for row in ordered
        )
        batches.append(SourceBatch("TREASURY", available, True, (token,)))
    return batches, {
        "physical_rows": len(frame),
        "complete_rows": complete_count,
        "incomplete_rows": incomplete_count,
        "batches": len(batches),
        "valid_batches": sum(batch.valid for batch in batches),
        "invalid_batches": sum(not batch.valid for batch in batches),
    }


def _step(current: Fraction, previous: Fraction) -> str:
    if current > previous:
        return "UP"
    if current < previous:
        return "DOWN"
    return "EQUAL"


def build_soma_batches(
    operations: pd.DataFrame,
    details: pd.DataFrame,
    *,
    enforce_physical_counts: bool = False,
) -> tuple[list[SourceBatch], dict[str, Any]]:
    if tuple(operations.columns) != prereg.SOMA_OPERATION_ALLOWLIST:
        raise RuntimeError("CLOR-D1 SOMA operation allowlist changed")
    if tuple(details.columns) != prereg.SOMA_DETAIL_ALLOWLIST:
        raise RuntimeError("CLOR-D1 SOMA detail allowlist changed")
    operation_map: dict[str, dict[str, Any]] = {}
    previous_date: date | None = None
    for raw in operations.to_dict(orient="records"):
        operation_id = raw["operation_id"]
        if not operation_id or operation_id in operation_map:
            raise RuntimeError("CLOR-D1 SOMA operation identity changed")
        operation_day = parse_date(raw["operation_date"])
        available = parse_source_time(raw["available_at_utc"])
        if not date(2019, 1, 1) <= operation_day <= date(2023, 12, 31):
            raise RuntimeError("CLOR-D1 SOMA date left frozen window")
        if previous_date is not None and operation_day < previous_date:
            raise RuntimeError("CLOR-D1 SOMA dates are not nondecreasing")
        previous_date = operation_day
        submitted = parse_fraction(
            raw["total_par_submitted"],
            nonnegative=True,
        )
        accepted = parse_fraction(
            raw["total_par_accepted"],
            nonnegative=True,
        )
        if (
            submitted is None
            or accepted is None
            or accepted > submitted
        ):
            raise RuntimeError("CLOR-D1 SOMA operation amounts changed")
        operation_map[operation_id] = {
            "date": operation_day,
            "available": available,
            "submitted": submitted,
            "accepted": accepted,
        }
    detail_sums: dict[str, list[Fraction | int]] = defaultdict(
        lambda: [Fraction(), Fraction(), 0]
    )
    for raw in details.to_dict(orient="records"):
        operation_id = raw["operation_id"]
        if not operation_id or operation_id not in operation_map:
            raise RuntimeError("CLOR-D1 SOMA detail operation is unknown")
        operation_day = parse_date(raw["operation_date"])
        available = parse_source_time(raw["available_at_utc"])
        if not date(2019, 1, 1) <= operation_day <= date(2023, 12, 31):
            raise RuntimeError("CLOR-D1 SOMA detail date left frozen window")
        operation = operation_map[operation_id]
        if (
            operation_day != operation["date"]
            or available != operation["available"]
        ):
            raise RuntimeError("CLOR-D1 SOMA detail identity disagrees")
        submitted = parse_fraction(raw["par_submitted"], nonnegative=True)
        accepted = parse_fraction(raw["par_accepted"], nonnegative=True)
        if (
            submitted is None
            or accepted is None
            or accepted > submitted
        ):
            raise RuntimeError("CLOR-D1 SOMA detail amounts changed")
        totals = detail_sums[operation_id]
        totals[0] = Fraction(totals[0]) + submitted
        totals[1] = Fraction(totals[1]) + accepted
        totals[2] = int(totals[2]) + 1
    for operation_id, operation in operation_map.items():
        totals = detail_sums.get(operation_id)
        if (
            totals is None
            or int(totals[2]) <= 0
            or totals[0] != operation["submitted"]
            or totals[1] != operation["accepted"]
        ):
            raise RuntimeError("CLOR-D1 SOMA detail reconciliation failed")
    if enforce_physical_counts and (
        len(operations) != 1_259 or len(details) != 182_616
    ):
        raise RuntimeError("CLOR-D1 SOMA physical counts changed")
    by_time: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for operation in operation_map.values():
        by_time[operation["available"]].append(operation)
    previous: tuple[Fraction, Fraction, Fraction] | None = None
    batches: list[SourceBatch] = []
    for available in sorted(by_time):
        submitted = sum(
            (row["submitted"] for row in by_time[available]),
            Fraction(),
        )
        accepted = sum(
            (row["accepted"] for row in by_time[available]),
            Fraction(),
        )
        if submitted <= 0 or accepted < 0 or accepted > submitted:
            raise RuntimeError("CLOR-D1 SOMA batch is structurally invalid")
        current = (submitted, accepted, accepted / submitted)
        if previous is not None:
            token = tuple(
                _step(current[index], previous[index]) for index in range(3)
            )
            batches.append(SourceBatch("SOMA", available, True, token))
        previous = current
    return batches, {
        "operation_rows": len(operations),
        "detail_rows": len(details),
        "operations": len(operation_map),
        "complete_batches": len(by_time),
        "transition_batches": len(batches),
    }


def _expected_ofr_availability(observation_day: date) -> datetime:
    delayed = datetime.combine(
        observation_day + timedelta(days=8),
        time(),
        UTC,
    )
    return max(
        delayed,
        datetime(2020, 9, 10, tzinfo=UTC),
    )


def build_ofr_batches(
    frame: pd.DataFrame,
    *,
    enforce_physical_counts: bool = False,
) -> tuple[list[SourceBatch], dict[str, Any]]:
    if tuple(frame.columns) != prereg.OFR_ALLOWLIST:
        raise RuntimeError("CLOR-D1 OFR allowlist changed")
    required = set(prereg.OFR_MNEMONICS)
    identities: set[tuple[str, date]] = set()
    all_dates: set[date] = set()
    selected: dict[date, dict[str, dict[str, Any]]] = defaultdict(dict)
    previous_date: date | None = None
    required_rows = 0
    for raw in frame.to_dict(orient="records"):
        mnemonic = raw["mnemonic"]
        observation_day = parse_date(raw["observation_date"])
        if not date(2019, 1, 1) <= observation_day <= date(2023, 12, 31):
            raise RuntimeError("CLOR-D1 OFR date left frozen window")
        if previous_date is not None and observation_day < previous_date:
            raise RuntimeError("CLOR-D1 OFR dates are not nondecreasing")
        previous_date = observation_day
        identity = (mnemonic, observation_day)
        if identity in identities:
            raise RuntimeError("CLOR-D1 OFR physical identity duplicated")
        identities.add(identity)
        if mnemonic not in required:
            continue
        all_dates.add(observation_day)
        required_rows += 1
        if mnemonic in selected[observation_day]:
            raise RuntimeError("CLOR-D1 OFR required row duplicated")
        available = parse_source_time(raw["available_at_utc"])
        if available != _expected_ofr_availability(observation_day):
            raise RuntimeError("CLOR-D1 OFR availability changed")
        if raw["disclosure_edit"] not in {"0", "1"}:
            raise RuntimeError("CLOR-D1 OFR disclosure marker changed")
        value = parse_fraction(
            raw["value"],
            allow_empty=True,
            nonnegative=("_TV_" in mnemonic),
        )
        selected[observation_day][mnemonic] = {
            "available": available,
            "value": value,
            "disclosure_edit": raw["disclosure_edit"],
        }
    if enforce_physical_counts and len(frame) != 77_369:
        raise RuntimeError("CLOR-D1 OFR physical row count changed")
    complete_by_time: dict[
        datetime,
        list[tuple[date, tuple[str, str] | None]],
    ] = defaultdict(list)
    complete_dates = incomplete_dates = 0
    for observation_day in sorted(all_dates):
        rows = selected.get(observation_day, {})
        available = _expected_ofr_availability(observation_day)
        complete = (
            set(rows) == required
            and all(
                row["disclosure_edit"] == "0" and row["value"] is not None
                for row in rows.values()
            )
        )
        if not complete:
            incomplete_dates += 1
            complete_by_time[available].append((observation_day, None))
            continue
        rate_values = {
            "DVP": rows["REPO-DVP_AR_TOT-P"]["value"],
            "GCF": rows["REPO-GCF_AR_TOT-P"]["value"],
            "TRIV1": rows["REPO-TRIV1_AR_TOT-P"]["value"],
        }
        volume_values = {
            "DVP": rows["REPO-DVP_TV_TOT-P"]["value"],
            "GCF": rows["REPO-GCF_TV_TOT-P"]["value"],
            "TRIV1": rows["REPO-TRIV1_TV_TOT-P"]["value"],
        }
        if any(value is None for value in (*rate_values.values(), *volume_values.values())):
            raise RuntimeError("CLOR-D1 OFR complete date lost a value")
        rates = {key: Fraction(value) for key, value in rate_values.items()}
        volumes = {
            key: Fraction(value) for key, value in volume_values.items()
        }
        if any(value < 0 for value in volumes.values()) or sum(
            volumes.values(),
            Fraction(),
        ) <= 0:
            raise RuntimeError("CLOR-D1 OFR volumes are invalid")
        token = (
            weak_order(rates, ("DVP", "GCF", "TRIV1")),
            weak_order(volumes, ("DVP", "GCF", "TRIV1")),
        )
        complete_dates += 1
        complete_by_time[available].append((observation_day, token))
    batches: list[SourceBatch] = []
    for available in sorted(complete_by_time):
        complete = [
            item for item in complete_by_time[available] if item[1] is not None
        ]
        if not complete:
            batches.append(SourceBatch("OFR", available, False))
            continue
        _, token = max(complete, key=lambda item: item[0])
        if token is None:
            raise RuntimeError("CLOR-D1 OFR complete selection failed")
        batches.append(SourceBatch("OFR", available, True, token))
    return batches, {
        "physical_rows": len(frame),
        "required_rows": required_rows,
        "physical_dates": len(all_dates),
        "complete_dates": complete_dates,
        "incomplete_dates": incomplete_dates,
        "batches": len(batches),
        "valid_batches": sum(batch.valid for batch in batches),
        "invalid_batches": sum(not batch.valid for batch in batches),
    }


def canonical_line(
    updated: Sequence[str],
    treasury: str,
    soma: Sequence[str],
    ofr: Sequence[str],
) -> str:
    if (
        not updated
        or any(source not in SOURCE_ORDER for source in updated)
        or tuple(updated)
        != tuple(source for source in SOURCE_ORDER if source in updated)
        or len(soma) != 3
        or len(ofr) != 2
    ):
        raise RuntimeError("CLOR-D1 canonical line inputs changed")
    return (
        f"UPDATED={','.join(updated)};"
        f"TREASURY={treasury};"
        f"SOMA={','.join(soma)};"
        f"OFR={','.join(ofr)}"
    )


def _invalid_reasons(
    states: Mapping[str, SourceState],
    current_time: datetime,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for source in SOURCE_ORDER:
        state = states.get(source)
        if state is None:
            reasons.append(f"MISSING_{source}")
        elif not state.valid:
            reasons.append(f"INVALID_{source}")
        elif current_time - state.available_at > FRESHNESS[source]:
            reasons.append(f"STALE_{source}")
    return tuple(reasons)


def build_joint_rows(batches: Sequence[SourceBatch]) -> list[JointRow]:
    ordered = sorted(
        batches,
        key=lambda batch: (
            execution_time(batch.available_at),
            batch.available_at,
            SOURCE_ORDER_INDEX[batch.source],
        ),
    )
    grouped: dict[datetime, list[SourceBatch]] = defaultdict(list)
    for batch in ordered:
        if batch.source not in SOURCE_ORDER:
            raise RuntimeError("CLOR-D1 source batch name changed")
        if batch.valid and not batch.token:
            raise RuntimeError("CLOR-D1 valid source batch has no token")
        if not batch.valid and batch.token:
            raise RuntimeError("CLOR-D1 invalid source batch has a token")
        grouped[execution_time(batch.available_at)].append(batch)
    states: dict[str, SourceState] = {}
    history: deque[str] = deque(maxlen=12)
    current_split: str | None = None
    output: list[JointRow] = []
    for current_time in sorted(grouped):
        group = sorted(
            grouped[current_time],
            key=lambda batch: (
                batch.available_at,
                SOURCE_ORDER_INDEX[batch.source],
            ),
        )
        updated = tuple(
            source
            for source in SOURCE_ORDER
            if any(batch.source == source for batch in group)
        )
        for batch in group:
            states[batch.source] = SourceState(
                available_at=batch.available_at,
                valid=batch.valid,
                token=batch.token,
            )
        split = split_for(current_time)
        if split is None:
            if current_time >= SPLITS["EVAL"][1]:
                current_split = None
                history.clear()
            continue
        if split != current_split:
            current_split = split
            history.clear()
        reasons = _invalid_reasons(states, current_time)
        if reasons:
            history.clear()
            output.append(
                JointRow(
                    split=split,
                    execution_time=current_time,
                    valid=False,
                    invalid_reason="|".join(reasons),
                    model_decision=False,
                    updated=updated,
                    treasury="",
                    soma_submitted_step="",
                    soma_accepted_step="",
                    soma_coverage_step="",
                    ofr_rate_order="",
                    ofr_volume_order="",
                    line_text="",
                    line_sha256="",
                    sequence_sha256="",
                    decision_expiry_time=None,
                )
            )
            continue
        treasury_state = states["TREASURY"].token
        soma_state = states["SOMA"].token
        ofr_state = states["OFR"].token
        if (
            len(treasury_state) != 1
            or len(soma_state) != 3
            or len(ofr_state) != 2
        ):
            raise RuntimeError("CLOR-D1 source-state token shape changed")
        line = canonical_line(
            updated,
            treasury_state[0],
            soma_state,
            ofr_state,
        )
        history.append(line)
        model_decision = len(history) == 12
        sequence_hash = (
            hashlib.sha256("\n".join(history).encode("ascii")).hexdigest()
            if model_decision
            else ""
        )
        output.append(
            JointRow(
                split=split,
                execution_time=current_time,
                valid=True,
                invalid_reason="",
                model_decision=model_decision,
                updated=updated,
                treasury=treasury_state[0],
                soma_submitted_step=soma_state[0],
                soma_accepted_step=soma_state[1],
                soma_coverage_step=soma_state[2],
                ofr_rate_order=ofr_state[0],
                ofr_volume_order=ofr_state[1],
                line_text=line,
                line_sha256=hashlib.sha256(
                    line.encode("ascii")
                ).hexdigest(),
                sequence_sha256=sequence_hash,
                decision_expiry_time=(
                    current_time + timedelta(hours=72)
                    if model_decision
                    else None
                ),
            )
        )
    return output


def weak_order_vocabulary(labels: Sequence[str]) -> tuple[str, ...]:
    if len(labels) != 3 or len(set(labels)) != 3:
        raise RuntimeError("CLOR-D1 random weak-order labels changed")
    vocabulary: set[str] = set()
    for ranks in itertools.product(range(3), repeat=3):
        used = set(ranks)
        if used != set(range(max(used) + 1)):
            continue
        groups = [
            "=".join(
                label
                for label, rank in zip(labels, ranks)
                if rank == level
            )
            for level in range(max(used) + 1)
        ]
        vocabulary.add(">".join(groups))
    result = tuple(sorted(vocabulary))
    if len(result) != 13:
        raise RuntimeError("CLOR-D1 random weak-order vocabulary changed")
    return result


def _hash_choice(key: str, vocabulary: Sequence[str]) -> str:
    digest = hashlib.sha256(key.encode("ascii")).digest()
    index = int.from_bytes(digest[:8], "big") % len(vocabulary)
    return vocabulary[index]


def _rotate_order(token: str, mapping: Mapping[str, str]) -> str:
    pieces = re.split(r"([>=])", token)
    return "".join(mapping.get(piece, piece) for piece in pieces)


def transform_batches(
    batches: Sequence[SourceBatch],
    control: str,
) -> list[SourceBatch]:
    if control not in RELATION_CONTROLS:
        raise RuntimeError(f"CLOR-D1 unknown control: {control}")
    transformed = list(batches)
    if control == "treasury_bidder_label_rotation":
        mapping = {"P": "D", "D": "I", "I": "P"}
        return [
            replace(
                batch,
                token=(
                    "|".join(
                        f"{term}:{_rotate_order(order, mapping)}"
                        for term, order in (
                            item.split(":", 1)
                            for item in batch.token[0].split("|")
                        )
                    ),
                ),
            )
            if batch.source == "TREASURY" and batch.valid
            else batch
            for batch in transformed
        ]
    if control == "soma_one_batch_stale":
        previous: tuple[str, ...] | None = None
        result: list[SourceBatch] = []
        for batch in sorted(
            transformed,
            key=lambda item: (
                item.available_at,
                SOURCE_ORDER_INDEX[item.source],
            ),
        ):
            if batch.source != "SOMA" or not batch.valid:
                result.append(batch)
                continue
            original = batch.token
            result.append(
                batch if previous is None else replace(batch, token=previous)
            )
            previous = original
        return result
    if control == "ofr_venue_label_rotation":
        mapping = {"DVP": "GCF", "GCF": "TRIV1", "TRIV1": "DVP"}
        return [
            replace(
                batch,
                token=tuple(
                    _rotate_order(token, mapping) for token in batch.token
                ),
            )
            if batch.source == "OFR" and batch.valid
            else batch
            for batch in transformed
        ]
    if control == "within_year_source_time_reverse":
        result = list(transformed)
        partitions: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, batch in enumerate(result):
            if batch.valid:
                partitions[(batch.source, batch.available_at.year)].append(
                    index
                )
        for indices in partitions.values():
            indices.sort(key=lambda index: result[index].available_at)
            reversed_tokens = [
                result[index].token for index in reversed(indices)
            ]
            for index, token in zip(indices, reversed_tokens):
                result[index] = replace(result[index], token=token)
        return result
    if control == "deterministic_random_relations":
        treasury_vocab = weak_order_vocabulary(("P", "D", "I"))
        ofr_vocab = weak_order_vocabulary(("DVP", "GCF", "TRIV1"))
        step_vocab = ("DOWN", "EQUAL", "UP")
        result = []
        for batch in transformed:
            if not batch.valid:
                result.append(batch)
                continue
            current_time = format_time(execution_time(batch.available_at))
            if batch.source == "TREASURY":
                terms = [
                    item.split(":", 1)[0] for item in batch.token[0].split("|")
                ]
                token = "|".join(
                    f"{term}:"
                    + _hash_choice(
                        (
                            "CLOR-D1|deterministic_random_relations|"
                            f"TREASURY|{current_time}|{term}"
                        ),
                        treasury_vocab,
                    )
                    for term in terms
                )
                result.append(replace(batch, token=(token,)))
            elif batch.source == "SOMA":
                fields = (
                    "submitted_step",
                    "accepted_step",
                    "coverage_step",
                )
                token = tuple(
                    _hash_choice(
                        (
                            "CLOR-D1|deterministic_random_relations|"
                            f"SOMA|{current_time}|{field}"
                        ),
                        step_vocab,
                    )
                    for field in fields
                )
                result.append(replace(batch, token=token))
            elif batch.source == "OFR":
                fields = ("rate_order", "volume_order")
                token = tuple(
                    _hash_choice(
                        (
                            "CLOR-D1|deterministic_random_relations|"
                            f"OFR|{current_time}|{field}"
                        ),
                        ofr_vocab,
                    )
                    for field in fields
                )
                result.append(replace(batch, token=token))
            else:
                raise RuntimeError("CLOR-D1 random relation source changed")
        return result
    if control == "one_merged_update_stale":
        return transformed
    raise RuntimeError(f"CLOR-D1 control not implemented: {control}")


def merged_update_stale_rows(primary: Sequence[JointRow]) -> list[JointRow]:
    output: list[JointRow] = []
    history: deque[str] = deque(maxlen=12)
    previous_primary: JointRow | None = None
    current_split: str | None = None
    for row in primary:
        if row.split != current_split:
            current_split = row.split
            history.clear()
            previous_primary = None
        if not row.valid:
            history.clear()
            previous_primary = None
            output.append(row)
            continue
        source = row if previous_primary is None else previous_primary
        line = canonical_line(
            source.updated,
            source.treasury,
            (
                source.soma_submitted_step,
                source.soma_accepted_step,
                source.soma_coverage_step,
            ),
            (source.ofr_rate_order, source.ofr_volume_order),
        )
        history.append(line)
        decision = len(history) == 12
        if decision != row.model_decision:
            raise RuntimeError("CLOR-D1 merged-stale schedule changed")
        output.append(
            JointRow(
                split=row.split,
                execution_time=row.execution_time,
                valid=True,
                invalid_reason="",
                model_decision=decision,
                updated=source.updated,
                treasury=source.treasury,
                soma_submitted_step=source.soma_submitted_step,
                soma_accepted_step=source.soma_accepted_step,
                soma_coverage_step=source.soma_coverage_step,
                ofr_rate_order=source.ofr_rate_order,
                ofr_volume_order=source.ofr_volume_order,
                line_text=line,
                line_sha256=hashlib.sha256(
                    line.encode("ascii")
                ).hexdigest(),
                sequence_sha256=(
                    hashlib.sha256(
                        "\n".join(history).encode("ascii")
                    ).hexdigest()
                    if decision
                    else ""
                ),
                decision_expiry_time=row.decision_expiry_time,
            )
        )
        previous_primary = row
    return output


def build_control_rows(
    batches: Sequence[SourceBatch],
    primary: Sequence[JointRow],
    control: str,
) -> list[JointRow]:
    if control == "one_merged_update_stale":
        return merged_update_stale_rows(primary)
    return build_joint_rows(transform_batches(batches, control))


def future_append_batches() -> tuple[SourceBatch, ...]:
    return (
        SourceBatch(
            "TREASURY",
            datetime(2024, 1, 2, 0, 0, tzinfo=UTC),
            True,
            ("2-Year:P>D>I",),
        ),
        SourceBatch(
            "SOMA",
            datetime(2024, 1, 2, 0, 5, tzinfo=UTC),
            True,
            ("UP", "UP", "UP"),
        ),
        SourceBatch(
            "OFR",
            datetime(2024, 1, 2, 0, 10, tzinfo=UTC),
            True,
            ("DVP>GCF>TRIV1", "TRIV1>GCF>DVP"),
        ),
    )


def build_source_batches(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    *,
    enforce_physical_counts: bool,
) -> tuple[list[SourceBatch], dict[str, Any]]:
    treasury, soma_operations, soma_details, ofr = frames
    treasury_batches, treasury_audit = build_treasury_batches(
        treasury,
        enforce_physical_counts=enforce_physical_counts,
    )
    soma_batches, soma_audit = build_soma_batches(
        soma_operations,
        soma_details,
        enforce_physical_counts=enforce_physical_counts,
    )
    ofr_batches, ofr_audit = build_ofr_batches(
        ofr,
        enforce_physical_counts=enforce_physical_counts,
    )
    batches = [*treasury_batches, *soma_batches, *ofr_batches]
    return batches, {
        "treasury": treasury_audit,
        "soma": soma_audit,
        "ofr": ofr_audit,
    }


def _decision_rows(rows: Sequence[JointRow]) -> list[JointRow]:
    return [row for row in rows if row.model_decision]


def _row_hash(rows: Sequence[JointRow]) -> str:
    return canonical_hash([row.csv_row() for row in rows])


def _gate_record(
    gate_id: int,
    checks: Mapping[str, bool],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if gate_id < 1 or gate_id > len(GATE_NAMES):
        raise RuntimeError("CLOR-D1 gate id is outside the frozen sequence")
    normalized = {str(key): bool(value) for key, value in checks.items()}
    return {
        "gate_id": gate_id,
        "name": GATE_NAMES[gate_id - 1],
        "checks": normalized,
        "metrics": dict(metrics),
        "passed": bool(normalized) and all(normalized.values()),
    }


def source_schema_gate(
    audits: Mapping[str, Any],
    first_batches: Sequence[SourceBatch],
    second_batches: Sequence[SourceBatch],
) -> dict[str, Any]:
    treasury = audits.get("treasury", {})
    soma = audits.get("soma", {})
    ofr = audits.get("ofr", {})
    checks = {
        "treasury_physical_rows": treasury.get("physical_rows") == 445,
        "treasury_complete_rows": treasury.get("complete_rows") == 440,
        "treasury_incomplete_rows": treasury.get("incomplete_rows") == 5,
        "soma_operation_rows": soma.get("operation_rows") == 1_259,
        "soma_detail_rows": soma.get("detail_rows") == 182_616,
        "soma_operations_reconciled": soma.get("operations") == 1_259,
        "ofr_physical_rows": ofr.get("physical_rows") == 77_369,
        "all_sources_emit_batches": all(
            any(batch.source == source for batch in first_batches)
            for source in SOURCE_ORDER
        ),
        "deterministic_batch_build": list(first_batches)
        == list(second_batches),
    }
    return _gate_record(
        1,
        checks,
        {
            "source_audit": dict(audits),
            "batch_count": len(first_batches),
            "valid_batch_count": sum(batch.valid for batch in first_batches),
            "invalid_batch_count": sum(
                not batch.valid for batch in first_batches
            ),
        },
    )


def _row_semantics(row: JointRow) -> bool:
    primitives = (
        row.treasury,
        row.soma_submitted_step,
        row.soma_accepted_step,
        row.soma_coverage_step,
        row.ofr_rate_order,
        row.ofr_volume_order,
        row.line_text,
        row.line_sha256,
    )
    if not row.updated:
        return False
    if row.updated != tuple(
        source for source in SOURCE_ORDER if source in row.updated
    ):
        return False
    if not row.valid:
        reasons = row.invalid_reason.split("|")
        return bool(
            row.invalid_reason
            and not row.model_decision
            and all(not value for value in primitives)
            and not row.sequence_sha256
            and row.decision_expiry_time is None
            and reasons
            == [
                reason
                for source in SOURCE_ORDER
                for reason in reasons
                if reason.endswith(f"_{source}")
            ]
            and all(
                reason
                in {
                    f"MISSING_{source}",
                    f"INVALID_{source}",
                    f"STALE_{source}",
                }
                for source in SOURCE_ORDER
                for reason in reasons
                if reason.endswith(f"_{source}")
            )
            and len(reasons) == len(
                {reason.rsplit("_", 1)[-1] for reason in reasons}
            )
        )
    if row.invalid_reason or any(not value for value in primitives):
        return False
    if hashlib.sha256(row.line_text.encode("ascii")).hexdigest() != row.line_sha256:
        return False
    if row.model_decision:
        return bool(
            HEX64.fullmatch(row.sequence_sha256)
            and row.decision_expiry_time
            == row.execution_time + timedelta(hours=72)
        )
    return not row.sequence_sha256 and row.decision_expiry_time is None


def _primary_schedule_checks(
    batches: Sequence[SourceBatch],
    rows: Sequence[JointRow],
) -> dict[str, bool]:
    groups: dict[datetime, list[SourceBatch]] = defaultdict(list)
    for batch in sorted(
        batches,
        key=lambda item: (
            execution_time(item.available_at),
            item.available_at,
            SOURCE_ORDER_INDEX[item.source],
        ),
    ):
        groups[execution_time(batch.available_at)].append(batch)
    expected_times = [
        current
        for current in sorted(groups)
        if split_for(current) is not None
    ]
    checks = {
        "one_row_per_research_execution_group": (
            [row.execution_time for row in rows] == expected_times
        ),
        "strictly_increasing_execution_time": all(
            left.execution_time < right.execution_time
            for left, right in zip(rows, rows[1:])
        ),
        "split_assignment_exact": all(
            split_for(row.execution_time) == row.split for row in rows
        ),
        "five_minute_execution_grid": all(
            row.execution_time.second == 0
            and row.execution_time.microsecond == 0
            and row.execution_time.minute % 5 == 0
            for row in rows
        ),
        "row_state_semantics": all(_row_semantics(row) for row in rows),
        "sealed_period_absent": all(
            row.execution_time < SPLITS["EVAL"][1] for row in rows
        ),
    }
    states: dict[str, SourceState] = {}
    history: deque[str] = deque(maxlen=12)
    current_split: str | None = None
    row_by_time = {row.execution_time: row for row in rows}
    causal = len(row_by_time) == len(rows)
    for current_time in sorted(groups):
        group = sorted(
            groups[current_time],
            key=lambda item: (
                item.available_at,
                SOURCE_ORDER_INDEX[item.source],
            ),
        )
        updated = tuple(
            source
            for source in SOURCE_ORDER
            if any(batch.source == source for batch in group)
        )
        for batch in group:
            states[batch.source] = SourceState(
                batch.available_at,
                batch.valid,
                batch.token,
            )
        split = split_for(current_time)
        if split is None:
            continue
        row = row_by_time.get(current_time)
        if row is None or row.updated != updated:
            causal = False
            continue
        if split != current_split:
            current_split = split
            history.clear()
        reasons = _invalid_reasons(states, current_time)
        if reasons:
            causal = causal and bool(
                not row.valid
                and row.invalid_reason == "|".join(reasons)
                and not row.model_decision
            )
            history.clear()
            continue
        treasury = states["TREASURY"].token
        soma = states["SOMA"].token
        ofr = states["OFR"].token
        expected_line = canonical_line(updated, treasury[0], soma, ofr)
        history.append(expected_line)
        expected_decision = len(history) == 12
        expected_sequence = (
            hashlib.sha256("\n".join(history).encode("ascii")).hexdigest()
            if expected_decision
            else ""
        )
        causal = causal and bool(
            row.valid
            and row.line_text == expected_line
            and row.model_decision == expected_decision
            and row.sequence_sha256 == expected_sequence
        )
    checks["causal_state_freshness_history"] = causal
    return checks


def causal_schedule_gate(
    batches: Sequence[SourceBatch],
    first_rows: Sequence[JointRow],
    second_rows: Sequence[JointRow],
) -> dict[str, Any]:
    appended = build_joint_rows([*batches, *future_append_batches()])
    checks = _primary_schedule_checks(batches, first_rows)
    checks.update(
        {
            "deterministic_primary_build": list(first_rows)
            == list(second_rows),
            "future_append_byte_invariant": [
                row.csv_row() for row in first_rows
            ]
            == [row.csv_row() for row in appended],
        }
    )
    return _gate_record(
        2,
        checks,
        {
            "source_rows": len(first_rows),
            "valid_rows": sum(row.valid for row in first_rows),
            "invalid_rows": sum(not row.valid for row in first_rows),
            "primary_row_hash": _row_hash(first_rows),
            "future_append_row_hash": _row_hash(appended),
        },
    )


def model_decision_count_gate(rows: Sequence[JointRow]) -> dict[str, Any]:
    counts = Counter(row.split for row in rows if row.model_decision)
    minimums = {"TRAIN": 450, "TEST": 180, "EVAL": 180}
    checks = {
        split: counts[split] >= minimum for split, minimum in minimums.items()
    }
    return _gate_record(
        3,
        checks,
        {
            "counts": {split: counts[split] for split in SPLITS},
            "minimums": minimums,
        },
    )


def source_update_support_gate(rows: Sequence[JointRow]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = {
        source: Counter() for source in SOURCE_ORDER
    }
    for row in rows:
        if not row.model_decision:
            continue
        csv_updated = row.csv_row()["updated"].split("|")
        for source in SOURCE_ORDER:
            if source in csv_updated:
                counts[source][row.split] += 1
    minimums = {
        "TREASURY": {"TRAIN": 40, "TEST": 20, "EVAL": 20},
        "SOMA": {"TRAIN": 200, "TEST": 90, "EVAL": 90},
        "OFR": {"TRAIN": 200, "TEST": 90, "EVAL": 90},
    }
    checks = {
        f"{source}:{split}": counts[source][split] >= minimum
        for source in SOURCE_ORDER
        for split, minimum in minimums[source].items()
    }
    return _gate_record(
        4,
        checks,
        {
            "counts": {
                source: {
                    split: counts[source][split] for split in SPLITS
                }
                for source in SOURCE_ORDER
            },
            "minimums": minimums,
        },
    )


def maximum_decision_gap_gate(rows: Sequence[JointRow]) -> dict[str, Any]:
    maximums: dict[str, int | None] = {}
    checks: dict[str, bool] = {}
    for split, (start, end) in SPLITS.items():
        decisions = sorted(
            row.execution_time
            for row in rows
            if row.split == split and row.model_decision
        )
        inside = bool(decisions) and all(start <= value < end for value in decisions)
        unique = len(decisions) == len(set(decisions))
        if decisions:
            endpoints = [start, *decisions, end]
            gaps = [
                (right - left).days * 86_400 + (right - left).seconds
                for left, right in zip(endpoints, endpoints[1:])
            ]
            maximums[split] = max(gaps)
        else:
            maximums[split] = None
        checks[f"{split}:inside_unique"] = inside and unique
        checks[f"{split}:maximum_le_864000"] = (
            maximums[split] is not None and maximums[split] <= 864_000
        )
    return _gate_record(
        5,
        checks,
        {
            "maximum_gap_seconds": maximums,
            "maximum_allowed_seconds": 864_000,
            "endpoint_inclusive": True,
        },
    )


def _quarter_start(year: int, quarter: int) -> datetime:
    return datetime(year, 1 + (quarter - 1) * 3, 1, tzinfo=UTC)


def calendar_support_gate(rows: Sequence[JointRow]) -> dict[str, Any]:
    decision_times = [
        row.execution_time for row in rows if row.model_decision
    ]
    windows: dict[str, tuple[datetime, datetime, int]] = {
        "2020_POST_FLOOR": (
            SPLITS["TRAIN"][0],
            datetime(2021, 1, 1, tzinfo=UTC),
            30,
        )
    }
    for quarter in range(1, 5):
        windows[f"2021Q{quarter}"] = (
            _quarter_start(2021, quarter),
            (
                _quarter_start(2021, quarter + 1)
                if quarter < 4
                else datetime(2022, 1, 1, tzinfo=UTC)
            ),
            50,
        )
        windows[f"2022Q{quarter}"] = (
            _quarter_start(2022, quarter),
            (
                _quarter_start(2022, quarter + 1)
                if quarter < 4
                else datetime(2023, 1, 1, tzinfo=UTC)
            ),
            40,
        )
        windows[f"2023Q{quarter}"] = (
            _quarter_start(2023, quarter),
            (
                _quarter_start(2023, quarter + 1)
                if quarter < 4
                else datetime(2024, 1, 1, tzinfo=UTC)
            ),
            40,
        )
    counts = {
        name: sum(start <= value < end for value in decision_times)
        for name, (start, end, _) in windows.items()
    }
    checks = {
        name: counts[name] >= minimum
        for name, (_, _, minimum) in windows.items()
    }
    return _gate_record(
        6,
        checks,
        {
            "counts": counts,
            "minimums": {
                name: minimum for name, (_, _, minimum) in windows.items()
            },
        },
    )


PRIMITIVE_FIELDS = (
    "treasury",
    "soma_submitted_step",
    "soma_accepted_step",
    "soma_coverage_step",
    "ofr_rate_order",
    "ofr_volume_order",
)


def primitive_diversity_gate(rows: Sequence[JointRow]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for split in SPLITS:
        split_rows = [row for row in rows if row.split == split and row.valid]
        metrics[split] = {}
        for field in PRIMITIVE_FIELDS:
            counts = Counter(getattr(row, field) for row in split_rows)
            total = sum(counts.values())
            maximum = max(counts.values(), default=0)
            metrics[split][field] = {
                "counts": dict(sorted(counts.items())),
                "levels": len(counts),
                "total": total,
                "maximum_level_count": maximum,
                "maximum_share_fraction": (
                    f"{maximum}/{total}" if total else None
                ),
            }
            checks[f"{split}:{field}:levels"] = (
                total > 0 and "" not in counts and len(counts) >= 2
            )
            checks[f"{split}:{field}:dominance"] = (
                total > 0 and maximum * 100 <= total * 95
            )
    return _gate_record(7, checks, metrics)


def state_signature_concentration_gate(
    rows: Sequence[JointRow],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for split in SPLITS:
        counts = Counter(
            row.line_text for row in rows if row.split == split and row.valid
        )
        total = sum(counts.values())
        maximum = max(counts.values(), default=0)
        metrics[split] = {
            "valid_lines": total,
            "unique_signatures": len(counts),
            "maximum_signature_count": maximum,
            "maximum_share_fraction": (
                f"{maximum}/{total}" if total else None
            ),
        }
        checks[split] = total > 0 and maximum * 4 <= total
    return _gate_record(8, checks, metrics)


def sequence_uniqueness_gate(rows: Sequence[JointRow]) -> dict[str, Any]:
    minimums = {"TRAIN": 150, "TEST": 70, "EVAL": 70}
    counts = {
        split: len(
            {
                row.sequence_sha256
                for row in rows
                if row.split == split and row.model_decision
            }
        )
        for split in SPLITS
    }
    checks = {
        split: counts[split] >= minimum for split, minimum in minimums.items()
    }
    return _gate_record(
        9,
        checks,
        {"counts": counts, "minimums": minimums},
    )


def _decision_hash_map(rows: Sequence[JointRow]) -> dict[datetime, str]:
    result: dict[datetime, str] = {}
    for row in rows:
        if not row.model_decision:
            continue
        if row.execution_time in result or not HEX64.fullmatch(
            row.sequence_sha256
        ):
            raise RuntimeError("CLOR-D1 decision timestamp/hash is invalid")
        result[row.execution_time] = row.sequence_sha256
    return result


def relation_controls_gate(
    primary: Sequence[JointRow],
    first_controls: Mapping[str, Sequence[JointRow]],
    second_controls: Mapping[str, Sequence[JointRow]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "control_id_set": tuple(first_controls) == tuple(RELATION_CONTROLS),
        "deterministic_control_id_set": tuple(second_controls)
        == tuple(RELATION_CONTROLS),
    }
    metrics: dict[str, Any] = {}
    primary_map = _decision_hash_map(primary)
    primary_schedule = [
        (
            row.split,
            row.execution_time,
            row.valid,
            row.model_decision,
        )
        for row in primary
    ]
    for control in RELATION_CONTROLS:
        first = list(first_controls.get(control, ()))
        second = list(second_controls.get(control, ()))
        deterministic = first == second
        schedule = [
            (
                row.split,
                row.execution_time,
                row.valid,
                row.model_decision,
            )
            for row in first
        ]
        schedule_aligned = schedule == primary_schedule
        try:
            control_map = _decision_hash_map(first)
            bijection = set(control_map) == set(primary_map)
        except RuntimeError:
            control_map = {}
            bijection = False
        eligible = len(primary_map) if bijection else 0
        changed = (
            sum(
                primary_map[timestamp] != control_map[timestamp]
                for timestamp in primary_map
            )
            if bijection
            else 0
        )
        vector_distinct = bool(
            bijection
            and canonical_hash(
                [
                    [format_time(timestamp), primary_map[timestamp]]
                    for timestamp in sorted(primary_map)
                ]
            )
            != canonical_hash(
                [
                    [format_time(timestamp), control_map[timestamp]]
                    for timestamp in sorted(control_map)
                ]
            )
        )
        metrics[control] = {
            "eligible_count": eligible,
            "changed_count": changed,
            "changed_fraction": (
                f"{changed}/{eligible}" if eligible else None
            ),
            "schedule_aligned": schedule_aligned,
            "vector_hash_distinct": vector_distinct,
        }
        checks[f"{control}:deterministic"] = deterministic
        checks[f"{control}:schedule_alignment"] = schedule_aligned
        checks[f"{control}:timestamp_bijection"] = bijection and eligible > 0
        checks[f"{control}:globally_hash_distinct"] = vector_distinct
        checks[f"{control}:changed_fraction"] = (
            eligible > 0 and changed * 10 >= eligible
        )
    return _gate_record(10, checks, metrics)


def forbidden_access_gate(
    counters: Mapping[str, int],
) -> dict[str, Any]:
    checks = {
        name: (
            set(counters) == set(FORBIDDEN_COUNTER_NAMES)
            and type(counters.get(name)) is int
            and counters[name] == 0
        )
        for name in FORBIDDEN_COUNTER_NAMES
    }
    return _gate_record(11, checks, {"counters": dict(counters)})


def _csv_bytes(
    records: Iterable[Mapping[str, str]],
    columns: Sequence[str],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(columns),
        extrasaction="raise",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for record in records:
        if set(record) != set(columns):
            raise RuntimeError("CLOR-D1 CSV record schema changed")
        writer.writerow({column: record[column] for column in columns})
    return stream.getvalue().encode("ascii")


def _deterministic_gzip(content: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=buffer,
        compresslevel=9,
        mtime=0,
    ) as stream:
        stream.write(content)
    return buffer.getvalue()


def deterministic_source_gzip(rows: Sequence[JointRow]) -> bytes:
    return _deterministic_gzip(
        _csv_bytes((row.csv_row() for row in rows), SOURCE_COLUMNS)
    )


def control_records(
    controls: Mapping[str, Sequence[JointRow]],
) -> list[dict[str, str]]:
    if tuple(controls) != tuple(RELATION_CONTROLS):
        raise RuntimeError("CLOR-D1 control order changed")
    records: list[dict[str, str]] = []
    for control in RELATION_CONTROLS:
        for row in controls[control]:
            records.append({"control": control, **row.csv_row()})
    return records


def deterministic_control_gzip(
    controls: Mapping[str, Sequence[JointRow]],
) -> bytes:
    return _deterministic_gzip(
        _csv_bytes(control_records(controls), CONTROL_COLUMNS)
    )


def _synthetic_batches() -> list[SourceBatch]:
    batches = [
        SourceBatch(
            "TREASURY",
            datetime(2020, 9, 9, 23, 45, tzinfo=UTC),
            True,
            ("2-Year:P>D>I",),
        ),
        SourceBatch(
            "SOMA",
            datetime(2020, 9, 9, 23, 45, tzinfo=UTC),
            True,
            ("EQUAL", "EQUAL", "EQUAL"),
        ),
        SourceBatch(
            "OFR",
            datetime(2020, 9, 9, 23, 45, tzinfo=UTC),
            True,
            ("DVP>GCF>TRIV1", "TRIV1>GCF>DVP"),
        ),
    ]
    treasury_orders = (
        "P>D>I",
        "D>I>P",
        "I=P>D",
        "P=I>D",
    )
    steps = ("UP", "DOWN", "EQUAL")
    ofr_orders = (
        "DVP>GCF>TRIV1",
        "GCF>TRIV1>DVP",
        "TRIV1=DVP>GCF",
        "DVP=GCF>TRIV1",
    )
    start = datetime(2020, 9, 10, 0, 0, tzinfo=UTC)
    for index in range(18):
        available = start + timedelta(minutes=5 * index)
        batches.extend(
            (
                SourceBatch(
                    "TREASURY",
                    available,
                    True,
                    (f"2-Year:{treasury_orders[index % 4]}",),
                ),
                SourceBatch(
                    "SOMA",
                    available,
                    True,
                    (
                        steps[index % 3],
                        steps[(index + 1) % 3],
                        steps[(index + 2) % 3],
                    ),
                ),
                SourceBatch(
                    "OFR",
                    available,
                    True,
                    (
                        ofr_orders[index % 4],
                        ofr_orders[(index + 1) % 4],
                    ),
                ),
            )
        )
    return batches


def build_self_check_manifest() -> dict[str, Any]:
    batches = _synthetic_batches()
    primary = build_joint_rows(batches)
    duplicate = build_joint_rows(batches)
    controls = {
        control: build_control_rows(batches, primary, control)
        for control in RELATION_CONTROLS
    }
    duplicate_controls = {
        control: build_control_rows(batches, duplicate, control)
        for control in RELATION_CONTROLS
    }
    schedule = causal_schedule_gate(batches, primary, duplicate)
    relation = relation_controls_gate(
        primary,
        controls,
        duplicate_controls,
    )
    source_gzip = deterministic_source_gzip(primary)
    control_gzip = deterministic_control_gzip(controls)
    checks = {
        "causal_schedule": schedule["passed"],
        "relation_controls": relation["passed"],
        "source_gzip_deterministic": (
            source_gzip == deterministic_source_gzip(primary)
        ),
        "control_gzip_deterministic": (
            control_gzip == deterministic_control_gzip(controls)
        ),
        "weak_order_vocabulary_size": (
            len(weak_order_vocabulary(("P", "D", "I"))) == 13
        ),
        "model_decision_schedule_nonempty": bool(_decision_rows(primary)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"CLOR-D1 synthetic self-check failed: {checks}")
    core = {
        "protocol_version": SELF_CHECK_PROTOCOL,
        "policy_id": POLICY_ID,
        "checks": checks,
        "synthetic": {
            "batch_count": len(batches),
            "source_row_count": len(primary),
            "model_decision_count": len(_decision_rows(primary)),
            "control_row_count": len(control_records(controls)),
            "source_row_hash": _row_hash(primary),
            "source_gzip_sha256": hashlib.sha256(source_gzip).hexdigest(),
            "control_gzip_sha256": hashlib.sha256(control_gzip).hexdigest(),
        },
        "source_value_rows_opened": 0,
        "predecessor_value_rows_opened": 0,
        "forbidden_access": forbidden_access(),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def self_check_bytes() -> bytes:
    payload = build_self_check_manifest()
    return canonical_bytes(payload) + b"\n"


def _run_self_check_subprocess() -> dict[str, Any]:
    argv = [sys.executable, RUNNER_PATH, "self-check"]
    completed = subprocess.run(
        argv,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("CLOR-D1 self-check subprocess failed")
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("CLOR-D1 self-check output is not JSON") from error
    if completed.stdout != canonical_bytes(payload) + b"\n":
        raise RuntimeError("CLOR-D1 self-check output is not canonical")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if (
        payload.get("protocol_version") != SELF_CHECK_PROTOCOL
        or payload.get("policy_id") != POLICY_ID
        or payload.get("manifest_hash") != canonical_hash(core)
        or payload.get("source_value_rows_opened") != 0
        or payload.get("predecessor_value_rows_opened") != 0
        or payload.get("forbidden_access") != forbidden_access()
    ):
        raise RuntimeError("CLOR-D1 self-check manifest mismatch")
    return {
        "argv": argv,
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "manifest_hash": payload["manifest_hash"],
        "source_value_rows_opened": 0,
        "predecessor_value_rows_opened": 0,
        "forbidden_access": forbidden_access(),
    }


def _pytest_summary(stdout: str, stderr: str) -> dict[str, int]:
    lines = [
        line.strip()
        for line in (stdout + "\n" + stderr).splitlines()
        if re.search(
            r"\b(?:passed|failed|skipped|errors?|xfailed|xpassed)\b",
            line,
        )
    ]
    if not lines:
        raise RuntimeError("CLOR-D1 pytest summary is absent")
    summary = lines[-1]
    counts = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    aliases = {"error": "errors", "errors": "errors"}
    for count, label in re.findall(
        r"([0-9]+)\s+(passed|failed|skipped|errors?|xfailed|xpassed)\b",
        summary,
    ):
        counts[aliases.get(label, label)] += int(count)
    if counts["passed"] <= 0:
        raise RuntimeError("CLOR-D1 pytest passed count is absent")
    return counts


def _run_pytest_verification() -> dict[str, Any]:
    argv = [".venv/bin/pytest", "-q", TEST_PATH]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."
    completed = subprocess.run(
        argv,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    counts = _pytest_summary(completed.stdout, completed.stderr)
    if (
        completed.returncode != 0
        or counts["failed"]
        or counts["skipped"]
        or counts["errors"]
        or counts["xfailed"]
        or counts["xpassed"]
    ):
        raise RuntimeError("CLOR-D1 exact pytest verification failed")
    return {
        "argv": argv,
        "environment_override": {"PYTHONPATH": "."},
        "exit_code": completed.returncode,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "errors": counts["errors"],
    }


def _implementation_binding(path: str) -> dict[str, str]:
    commit = _assert_committed(path)
    digest = sha256_file(path)
    if not HEX40.fullmatch(commit) or not HEX64.fullmatch(digest):
        raise RuntimeError("CLOR-D1 implementation binding is malformed")
    if _git_blob_sha256(commit, path) != digest:
        raise RuntimeError("CLOR-D1 implementation blob differs")
    return _binding(path, commit, digest)


def build_execution_seal() -> dict[str, Any]:
    if not _worktree_clean():
        raise RuntimeError("CLOR-D1 seal creation requires a clean worktree")
    authority = static_authority()
    runner = _implementation_binding(RUNNER_PATH)
    tests = _implementation_binding(TEST_PATH)
    head = _git_output("rev-parse", "HEAD")
    if runner["commit"] != tests["commit"] or runner["commit"] != head:
        raise RuntimeError(
            "CLOR-D1 runner/tests must share the current HEAD commit"
        )
    self_check = _run_self_check_subprocess()
    pytest_verification = _run_pytest_verification()
    core = {
        "protocol_version": SEAL_PROTOCOL,
        "policy_id": POLICY_ID,
        "runtime": authority["runtime"],
        "contract": authority["contract"],
        "boundary": authority["boundary"],
        "preregistration": authority["preregistration"],
        "preregistration_producer": authority[
            "preregistration_producer"
        ],
        "runner": runner,
        "tests": tests,
        "shared_commit": runner["commit"],
        "synthetic_verification": {
            "self_check": self_check,
            "pytest": pytest_verification,
        },
        "forbidden_access": forbidden_access(),
    }
    return {**core, "seal_hash": canonical_hash(core)}


def create_execution_seal() -> dict[str, Any]:
    payload = build_execution_seal()
    prereg._write_once_bytes(EXECUTION_SEAL_PATH, json_bytes(payload))
    return payload


def _validate_exact_binding(
    binding: Mapping[str, Any],
    *,
    path: str,
    expected_commit: str | None = None,
    expected_sha256: str | None = None,
) -> None:
    if set(binding) != {"path", "commit", "sha256"}:
        raise RuntimeError("CLOR-D1 seal binding schema mismatch")
    commit = binding.get("commit")
    digest = binding.get("sha256")
    if (
        binding.get("path") != path
        or not isinstance(commit, str)
        or not HEX40.fullmatch(commit)
        or not isinstance(digest, str)
        or not HEX64.fullmatch(digest)
        or (expected_commit is not None and commit != expected_commit)
        or (expected_sha256 is not None and digest != expected_sha256)
    ):
        raise RuntimeError("CLOR-D1 seal binding value mismatch")
    _validate_binding(path, commit, digest)


def validate_execution_seal() -> dict[str, Any]:
    path = repository_path(EXECUTION_SEAL_PATH)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("CLOR-D1 execution seal is absent or unsafe")
    _assert_committed(EXECUTION_SEAL_PATH)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("CLOR-D1 execution seal is unreadable") from error
    if raw != json_bytes(payload):
        raise RuntimeError("CLOR-D1 execution seal bytes are noncanonical")
    expected_keys = {
        "protocol_version",
        "policy_id",
        "runtime",
        "contract",
        "boundary",
        "preregistration",
        "preregistration_producer",
        "runner",
        "tests",
        "shared_commit",
        "synthetic_verification",
        "forbidden_access",
        "seal_hash",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("CLOR-D1 execution seal schema mismatch")
    core = {key: value for key, value in payload.items() if key != "seal_hash"}
    if (
        payload.get("protocol_version") != SEAL_PROTOCOL
        or payload.get("policy_id") != POLICY_ID
        or payload.get("seal_hash") != canonical_hash(core)
        or payload.get("forbidden_access") != forbidden_access()
    ):
        raise RuntimeError("CLOR-D1 execution seal core mismatch")
    authority = static_authority()
    for key in (
        "runtime",
        "contract",
        "boundary",
        "preregistration",
        "preregistration_producer",
    ):
        if payload.get(key) != authority[key]:
            raise RuntimeError(f"CLOR-D1 execution seal {key} drift")
    shared_commit = payload.get("shared_commit")
    if not isinstance(shared_commit, str) or not HEX40.fullmatch(shared_commit):
        raise RuntimeError("CLOR-D1 execution seal shared commit malformed")
    _validate_exact_binding(
        payload["runner"],
        path=RUNNER_PATH,
        expected_commit=shared_commit,
    )
    _validate_exact_binding(
        payload["tests"],
        path=TEST_PATH,
        expected_commit=shared_commit,
    )
    ancestry = prereg._run_git(
        "merge-base",
        "--is-ancestor",
        shared_commit,
        _git_output("rev-parse", "HEAD"),
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError("CLOR-D1 sealed implementation is not an ancestor")
    verification = payload.get("synthetic_verification")
    if (
        not isinstance(verification, dict)
        or set(verification) != {"self_check", "pytest"}
        or verification["self_check"] != _run_self_check_subprocess()
        or verification["pytest"] != _run_pytest_verification()
    ):
        raise RuntimeError("CLOR-D1 sealed synthetic verification drift")
    return payload


def _schedule_funnel(rows: Sequence[JointRow]) -> dict[str, Any]:
    return {
        split: {
            "rows": sum(row.split == split for row in rows),
            "valid_rows": sum(
                row.split == split and row.valid for row in rows
            ),
            "invalid_rows": sum(
                row.split == split and not row.valid for row in rows
            ),
            "model_decisions": sum(
                row.split == split and row.model_decision for row in rows
            ),
        }
        for split in SPLITS
    }


def _authority_report(
    seal: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "execution_seal": {
            "path": EXECUTION_SEAL_PATH,
            "sha256": sha256_file(EXECUTION_SEAL_PATH),
            "seal_hash": seal["seal_hash"],
            "shared_commit": seal["shared_commit"],
            "runner": seal["runner"],
            "tests": seal["tests"],
        },
        "runtime": authority["runtime"],
        "contract": authority["contract"],
        "boundary": authority["boundary"],
        "preregistration": authority["preregistration"],
        "preregistration_producer": authority[
            "preregistration_producer"
        ],
        "source_authority_hash": authority["source_authority_hash"],
    }


def build_result_report(
    *,
    decision: str,
    authority: Mapping[str, Any],
    ledger: AccessLedger,
    audits: Mapping[str, Any],
    primary: Sequence[JointRow],
    controls: Mapping[str, Sequence[JointRow]] | None,
    gates: Sequence[Mapping[str, Any]],
    counters: Mapping[str, int],
    artifacts: Mapping[str, Any] | None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    if decision not in {"pass", "reject"}:
        raise RuntimeError("CLOR-D1 result decision changed")
    first_failure = next(
        (
            {
                "gate_id": gate["gate_id"],
                "name": gate["name"],
            }
            for gate in gates
            if not gate["passed"]
        ),
        None,
    )
    control_hashes = (
        None
        if controls is None
        else {
            control: _row_hash(controls[control])
            for control in RELATION_CONTROLS
        }
    )
    core = {
        "protocol_version": RESULT_PROTOCOL,
        "policy_id": POLICY_ID,
        "decision": decision,
        "terminal_action": (
            PASS_ACTION if decision == "pass" else FAILURE_ACTION
        ),
        "profitability_result": False,
        "outcomes_opened": False,
        "source_values_opened": any(ledger.decoded_rows().values()),
        "authority": dict(authority),
        "decoded_rows": ledger.decoded_rows(),
        "source_audit": dict(audits),
        "schedule_funnel": _schedule_funnel(primary),
        "gates": list(gates),
        "first_failure": first_failure,
        "artifacts": None if artifacts is None else dict(artifacts),
        "canonical_row_hashes": {
            "source": _row_hash(primary) if primary else None,
            "controls": control_hashes,
        },
        "forbidden_access": dict(counters),
        "error": (
            None
            if error is None
            else {"type": type(error).__name__, "message": str(error)}
        ),
    }
    if decision == "pass":
        if (
            len(gates) != len(GATE_NAMES)
            or first_failure is not None
            or artifacts is None
            or controls is None
        ):
            raise RuntimeError("CLOR-D1 pass report is incomplete")
    elif not gates or first_failure is None or artifacts is not None:
        raise RuntimeError("CLOR-D1 rejection report is incomplete")
    return {**core, "result_hash": canonical_hash(core)}


def _read_regular(path: str) -> bytes:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"CLOR-D1 terminal path is unsafe: {path}")
    return target.read_bytes()


def _gzip_csv_records(
    content: bytes,
    columns: Sequence[str],
) -> list[dict[str, str]]:
    if (
        len(content) < 10
        or content[:3] != b"\x1f\x8b\x08"
        or content[3] & 0x08
        or int.from_bytes(content[4:8], "little") != 0
    ):
        raise RuntimeError("CLOR-D1 deterministic gzip header changed")
    try:
        text = gzip.decompress(content).decode("ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise RuntimeError("CLOR-D1 terminal gzip is unreadable") from error
    if "\r" in text or not text.endswith("\n"):
        raise RuntimeError("CLOR-D1 terminal CSV newline changed")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != tuple(columns):
        raise RuntimeError("CLOR-D1 terminal CSV schema changed")
    records = list(reader)
    if any(set(record) != set(columns) or None in record.values() for record in records):
        raise RuntimeError("CLOR-D1 terminal CSV record changed")
    return [{column: record[column] for column in columns} for record in records]


def _validate_result_report(
    payload: Mapping[str, Any],
    *,
    decision: str,
) -> None:
    expected_keys = {
        "protocol_version",
        "policy_id",
        "decision",
        "terminal_action",
        "profitability_result",
        "outcomes_opened",
        "source_values_opened",
        "authority",
        "decoded_rows",
        "source_audit",
        "schedule_funnel",
        "gates",
        "first_failure",
        "artifacts",
        "canonical_row_hashes",
        "forbidden_access",
        "error",
        "result_hash",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("CLOR-D1 terminal report schema mismatch")
    core = {key: value for key, value in payload.items() if key != "result_hash"}
    if (
        payload.get("protocol_version") != RESULT_PROTOCOL
        or payload.get("policy_id") != POLICY_ID
        or payload.get("decision") != decision
        or payload.get("result_hash") != canonical_hash(core)
        or payload.get("profitability_result") is not False
        or payload.get("outcomes_opened") is not False
        or payload.get("forbidden_access") != forbidden_access()
    ):
        raise RuntimeError("CLOR-D1 terminal report core mismatch")
    gates = payload.get("gates")
    if not isinstance(gates, list) or not gates:
        raise RuntimeError("CLOR-D1 terminal report gate list is absent")
    for index, gate in enumerate(gates, start=1):
        if (
            not isinstance(gate, dict)
            or gate.get("gate_id") != index
            or gate.get("name") != GATE_NAMES[index - 1]
            or gate.get("passed")
            != (
                bool(gate.get("checks"))
                and all(gate.get("checks", {}).values())
            )
        ):
            raise RuntimeError("CLOR-D1 terminal gate record mismatch")
    if decision == "reject":
        if (
            payload.get("terminal_action") != FAILURE_ACTION
            or payload.get("artifacts") is not None
            or len(gates) > len(GATE_NAMES)
            or any(not gate["passed"] for gate in gates[:-1])
            or gates[-1]["passed"]
            or payload.get("first_failure")
            != {
                "gate_id": gates[-1]["gate_id"],
                "name": gates[-1]["name"],
            }
        ):
            raise RuntimeError("CLOR-D1 rejection gate prefix mismatch")
        return
    if (
        payload.get("terminal_action") != PASS_ACTION
        or len(gates) != len(GATE_NAMES)
        or not all(gate["passed"] for gate in gates)
        or payload.get("first_failure") is not None
        or not isinstance(payload.get("artifacts"), dict)
    ):
        raise RuntimeError("CLOR-D1 pass gate sequence mismatch")


def _csv_boolean(value: str) -> bool:
    if value not in {"0", "1"}:
        raise RuntimeError("CLOR-D1 terminal boolean changed")
    return value == "1"


def _joint_row_from_csv(record: Mapping[str, str]) -> JointRow:
    if set(record) != set(SOURCE_COLUMNS):
        raise RuntimeError("CLOR-D1 terminal source row schema changed")
    split = record["split"]
    if split not in SPLITS:
        raise RuntimeError("CLOR-D1 terminal split changed")
    execution = parse_canonical_time(record["execution_time"])
    expiry = (
        None
        if record["decision_expiry_time"] == ""
        else parse_canonical_time(record["decision_expiry_time"])
    )
    updated = tuple(record["updated"].split("|")) if record["updated"] else ()
    row = JointRow(
        split=split,
        execution_time=execution,
        valid=_csv_boolean(record["valid"]),
        invalid_reason=record["invalid_reason"],
        model_decision=_csv_boolean(record["model_decision"]),
        updated=updated,
        treasury=record["treasury"],
        soma_submitted_step=record["soma_submitted_step"],
        soma_accepted_step=record["soma_accepted_step"],
        soma_coverage_step=record["soma_coverage_step"],
        ofr_rate_order=record["ofr_rate_order"],
        ofr_volume_order=record["ofr_volume_order"],
        line_text=record["line_text"],
        line_sha256=record["line_sha256"],
        sequence_sha256=record["sequence_sha256"],
        decision_expiry_time=expiry,
    )
    if row.csv_row() != dict(record):
        raise RuntimeError("CLOR-D1 terminal source row is noncanonical")
    return row


def _artifact_schedule_semantics(rows: Sequence[JointRow]) -> bool:
    if not rows or not all(_row_semantics(row) for row in rows):
        return False
    if any(
        left.execution_time >= right.execution_time
        for left, right in zip(rows, rows[1:])
    ):
        return False
    history: deque[str] = deque(maxlen=12)
    current_split: str | None = None
    for row in rows:
        if split_for(row.execution_time) != row.split:
            return False
        if row.split != current_split:
            current_split = row.split
            history.clear()
        if not row.valid:
            history.clear()
            continue
        expected_line = canonical_line(
            row.updated,
            row.treasury,
            (
                row.soma_submitted_step,
                row.soma_accepted_step,
                row.soma_coverage_step,
            ),
            (row.ofr_rate_order, row.ofr_volume_order),
        )
        if row.line_text != expected_line:
            return False
        history.append(expected_line)
        expected_decision = len(history) == 12
        expected_hash = (
            hashlib.sha256("\n".join(history).encode("ascii")).hexdigest()
            if expected_decision
            else ""
        )
        if (
            row.model_decision != expected_decision
            or row.sequence_sha256 != expected_hash
        ):
            return False
    return True


def _validate_pass_source_audit(payload: Mapping[str, Any]) -> None:
    audit = payload.get("source_audit", {})
    treasury = audit.get("treasury", {})
    soma = audit.get("soma", {})
    ofr = audit.get("ofr", {})
    decoded = payload.get("decoded_rows")
    if (
        treasury.get("physical_rows") != 445
        or treasury.get("complete_rows") != 440
        or treasury.get("incomplete_rows") != 5
        or soma.get("operation_rows") != 1_259
        or soma.get("detail_rows") != 182_616
        or soma.get("operations") != 1_259
        or ofr.get("physical_rows") != 77_369
        or decoded
        != {
            "treasury": 445,
            "soma_operations": 1_259,
            "soma_details": 182_616,
            "ofr": 77_369,
            "predecessors": 0,
        }
        or payload.get("source_values_opened") is not True
    ):
        raise RuntimeError("CLOR-D1 pass source audit is inconsistent")


def _validate_pass_artifacts(payload: Mapping[str, Any]) -> None:
    artifacts = payload["artifacts"]
    if set(artifacts) != {"source", "controls"}:
        raise RuntimeError("CLOR-D1 pass artifact schema mismatch")
    source_bytes = _read_regular(SOURCE_OUTPUT)
    control_bytes = _read_regular(CONTROL_OUTPUT)
    source_records = _gzip_csv_records(source_bytes, SOURCE_COLUMNS)
    control_rows = _gzip_csv_records(control_bytes, CONTROL_COLUMNS)
    expected = {
        "source": {
            "path": SOURCE_OUTPUT,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "rows": len(source_records),
            "row_hash": canonical_hash(source_records),
        },
        "controls": {
            "path": CONTROL_OUTPUT,
            "sha256": hashlib.sha256(control_bytes).hexdigest(),
            "rows": len(control_rows),
            "row_hash": canonical_hash(control_rows),
        },
    }
    if artifacts != expected:
        raise RuntimeError("CLOR-D1 pass artifact hash/count drift")
    row_hashes = payload.get("canonical_row_hashes", {})
    if (
        row_hashes.get("source") != expected["source"]["row_hash"]
        or not isinstance(row_hashes.get("controls"), dict)
    ):
        raise RuntimeError("CLOR-D1 pass source row hash drift")
    by_control: dict[str, list[dict[str, str]]] = {
        control: [] for control in RELATION_CONTROLS
    }
    observed_order: list[str] = []
    for record in control_rows:
        control = record["control"]
        if control not in by_control:
            raise RuntimeError("CLOR-D1 terminal control id changed")
        if not observed_order or observed_order[-1] != control:
            observed_order.append(control)
        by_control[control].append(
            {column: record[column] for column in SOURCE_COLUMNS}
        )
    if tuple(observed_order) != tuple(RELATION_CONTROLS):
        raise RuntimeError("CLOR-D1 terminal control order changed")
    if row_hashes["controls"] != {
        control: canonical_hash(by_control[control])
        for control in RELATION_CONTROLS
    }:
        raise RuntimeError("CLOR-D1 terminal control row hash drift")
    primary = [_joint_row_from_csv(record) for record in source_records]
    controls = {
        control: [
            _joint_row_from_csv(record) for record in by_control[control]
        ]
        for control in RELATION_CONTROLS
    }
    if (
        deterministic_source_gzip(primary) != source_bytes
        or deterministic_control_gzip(controls) != control_bytes
        or not _artifact_schedule_semantics(primary)
        or not all(
            _artifact_schedule_semantics(controls[control])
            for control in RELATION_CONTROLS
        )
    ):
        raise RuntimeError("CLOR-D1 pass artifact semantics changed")
    recomputed = [
        model_decision_count_gate(primary),
        source_update_support_gate(primary),
        maximum_decision_gap_gate(primary),
        calendar_support_gate(primary),
        primitive_diversity_gate(primary),
        state_signature_concentration_gate(primary),
        sequence_uniqueness_gate(primary),
        relation_controls_gate(primary, controls, controls),
        forbidden_access_gate(payload["forbidden_access"]),
    ]
    if payload["gates"][2:] != recomputed:
        raise RuntimeError("CLOR-D1 pass gate evidence is not reproducible")
    first_gate, second_gate = payload["gates"][:2]
    if (
        not first_gate["passed"]
        or not second_gate["passed"]
        or not all(first_gate["checks"].values())
        or not all(second_gate["checks"].values())
        or second_gate["metrics"].get("primary_row_hash")
        != canonical_hash(source_records)
        or payload.get("schedule_funnel") != _schedule_funnel(primary)
    ):
        raise RuntimeError("CLOR-D1 pass source/schedule evidence drift")
    _validate_pass_source_audit(payload)


def _canonical_report(path: str) -> dict[str, Any]:
    raw = _read_regular(path)
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("CLOR-D1 terminal report unreadable") from error
    if raw != json_bytes(payload):
        raise RuntimeError("CLOR-D1 terminal report bytes are noncanonical")
    if not isinstance(payload, dict):
        raise RuntimeError("CLOR-D1 terminal report is not an object")
    return payload


def terminal_state() -> dict[str, Any] | None:
    paths = (SOURCE_OUTPUT, CONTROL_OUTPUT, PASS_REPORT, REJECTION_REPORT)
    exists = {path: prereg._output_entry_exists(path) for path in paths}
    if not any(exists.values()):
        return None
    if exists[REJECTION_REPORT] and not any(
        exists[path] for path in (SOURCE_OUTPUT, CONTROL_OUTPUT, PASS_REPORT)
    ):
        payload = _canonical_report(REJECTION_REPORT)
        _validate_result_report(payload, decision="reject")
        return dict(payload)
    if (
        not exists[REJECTION_REPORT]
        and all(
            exists[path] for path in (SOURCE_OUTPUT, CONTROL_OUTPUT, PASS_REPORT)
        )
    ):
        payload = _canonical_report(PASS_REPORT)
        _validate_result_report(payload, decision="pass")
        _validate_pass_artifacts(payload)
        return dict(payload)
    raise RuntimeError("CLOR-D1 terminal state is partial or conflicting")


def _publish_rejection(payload: Mapping[str, Any]) -> None:
    _validate_result_report(payload, decision="reject")
    if any(
        prereg._output_entry_exists(path)
        for path in (SOURCE_OUTPUT, CONTROL_OUTPUT, PASS_REPORT)
    ):
        raise RuntimeError("CLOR-D1 pass artifact exists before rejection")
    prereg._write_once_bytes(REJECTION_REPORT, json_bytes(payload))


def _publish_pass_group(
    source_bytes: bytes,
    control_bytes: bytes,
    report: Mapping[str, Any],
) -> None:
    _validate_result_report(report, decision="pass")
    if any(
        prereg._output_entry_exists(path)
        for path in (
            SOURCE_OUTPUT,
            CONTROL_OUTPUT,
            PASS_REPORT,
            REJECTION_REPORT,
        )
    ):
        raise RuntimeError("CLOR-D1 terminal target already exists")
    entries = (
        (SOURCE_OUTPUT, source_bytes),
        (CONTROL_OUTPUT, control_bytes),
        (PASS_REPORT, json_bytes(report)),
    )
    staged: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for relative, content in entries:
            target = repository_path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            parent_descriptor, _ = prereg._open_output_parent(relative)
            os.close(parent_descriptor)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".clor-stage",
                dir=target.parent,
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((temporary, target))
        for temporary, target in staged:
            os.link(temporary, target)
            published.append((temporary, target))
        for parent in {target.parent for _, target in staged}:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        for temporary, target in reversed(published):
            try:
                if target.exists() and os.path.samefile(temporary, target):
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def run_official() -> dict[str, Any]:
    existing = terminal_state()
    if existing is not None:
        seal = validate_execution_seal()
        authority = static_authority()
        if existing.get("authority") != _authority_report(seal, authority):
            raise RuntimeError("CLOR-D1 terminal authority drift")
        return existing

    authority = static_authority()
    seal = validate_execution_seal()
    authority_record = _authority_report(seal, authority)
    if not _worktree_clean():
        raise RuntimeError("CLOR-D1 official run requires a clean worktree")
    if forbidden_access() != {name: 0 for name in FORBIDDEN_COUNTER_NAMES}:
        raise RuntimeError("CLOR-D1 forbidden counter precondition failed")
    if terminal_state() is not None:
        raise RuntimeError("CLOR-D1 terminal state appeared concurrently")

    ledger = AccessLedger()
    audits: dict[str, Any] = {}
    primary: list[JointRow] = []
    controls: dict[str, list[JointRow]] | None = None
    gates: list[dict[str, Any]] = []
    counters = forbidden_access()

    def reject(error: BaseException | None = None) -> dict[str, Any]:
        report = build_result_report(
            decision="reject",
            authority=authority_record,
            ledger=ledger,
            audits=audits,
            primary=primary,
            controls=controls,
            gates=gates,
            counters=counters,
            artifacts=None,
            error=error,
        )
        _publish_rejection(report)
        return report

    try:
        frames = load_source_frames(ledger)
        batches, audits = build_source_batches(
            frames,
            enforce_physical_counts=True,
        )
        duplicate_batches, duplicate_audits = build_source_batches(
            frames,
            enforce_physical_counts=True,
        )
        gate = source_schema_gate(audits, batches, duplicate_batches)
        gate["checks"]["deterministic_audit_build"] = (
            audits == duplicate_audits
        )
        gate["passed"] = all(gate["checks"].values())
    except Exception as error:
        gates.append(
            _gate_record(
                1,
                {"source_build_completed": False},
                {
                    "decoded_rows": ledger.decoded_rows(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
        )
        return reject(error)
    gates.append(gate)
    if not gate["passed"]:
        return reject()

    try:
        primary = build_joint_rows(batches)
        duplicate_primary = build_joint_rows(duplicate_batches)
        gate = causal_schedule_gate(
            batches,
            primary,
            duplicate_primary,
        )
    except Exception as error:
        gates.append(
            _gate_record(
                2,
                {"source_schedule_build_completed": False},
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
        )
        return reject(error)
    gates.append(gate)
    if not gate["passed"]:
        return reject()

    simple_gates = (
        (3, lambda: model_decision_count_gate(primary)),
        (4, lambda: source_update_support_gate(primary)),
        (5, lambda: maximum_decision_gap_gate(primary)),
        (6, lambda: calendar_support_gate(primary)),
        (7, lambda: primitive_diversity_gate(primary)),
        (8, lambda: state_signature_concentration_gate(primary)),
        (9, lambda: sequence_uniqueness_gate(primary)),
    )
    for gate_id, builder in simple_gates:
        try:
            gate = builder()
        except Exception as error:
            gates.append(
                _gate_record(
                    gate_id,
                    {"gate_evaluation_completed": False},
                    {
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )
            )
            return reject(error)
        gates.append(gate)
        if not gate["passed"]:
            return reject()

    try:
        controls = {
            control: build_control_rows(batches, primary, control)
            for control in RELATION_CONTROLS
        }
        duplicate_controls = {
            control: build_control_rows(
                duplicate_batches,
                duplicate_primary,
                control,
            )
            for control in RELATION_CONTROLS
        }
        gate = relation_controls_gate(
            primary,
            controls,
            duplicate_controls,
        )
    except Exception as error:
        gates.append(
            _gate_record(
                10,
                {"control_build_completed": False},
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
        )
        return reject(error)
    gates.append(gate)
    if not gate["passed"]:
        return reject()

    gate = forbidden_access_gate(counters)
    gates.append(gate)
    if not gate["passed"]:
        return reject()
    if controls is None:
        raise RuntimeError("CLOR-D1 controls disappeared after Gate 10")

    source_bytes = deterministic_source_gzip(primary)
    control_bytes = deterministic_control_gzip(controls)
    source_records = [row.csv_row() for row in primary]
    controls_records = control_records(controls)
    artifacts = {
        "source": {
            "path": SOURCE_OUTPUT,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "rows": len(source_records),
            "row_hash": canonical_hash(source_records),
        },
        "controls": {
            "path": CONTROL_OUTPUT,
            "sha256": hashlib.sha256(control_bytes).hexdigest(),
            "rows": len(controls_records),
            "row_hash": canonical_hash(controls_records),
        },
    }
    report = build_result_report(
        decision="pass",
        authority=authority_record,
        ledger=ledger,
        audits=audits,
        primary=primary,
        controls=controls,
        gates=gates,
        counters=counters,
        artifacts=artifacts,
    )
    _publish_pass_group(source_bytes, control_bytes, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-check")
    subparsers.add_parser("create-seal")
    subparsers.add_parser("validate-seal")
    subparsers.add_parser("run")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.command == "self-check":
        sys.stdout.buffer.write(self_check_bytes())
        return
    if arguments.command == "create-seal":
        payload = create_execution_seal()
        print(
            json.dumps(
                {
                    "path": EXECUTION_SEAL_PATH,
                    "seal_hash": payload["seal_hash"],
                    "shared_commit": payload["shared_commit"],
                    "forbidden_access": payload["forbidden_access"],
                },
                sort_keys=True,
                indent=2,
            )
        )
        return
    if arguments.command == "validate-seal":
        payload = validate_execution_seal()
        print(
            json.dumps(
                {
                    "path": EXECUTION_SEAL_PATH,
                    "seal_hash": payload["seal_hash"],
                },
                sort_keys=True,
                indent=2,
            )
        )
        return
    report = run_official()
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "terminal_action": report["terminal_action"],
                "first_failure": report["first_failure"],
                "result_hash": report["result_hash"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    if report["decision"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
