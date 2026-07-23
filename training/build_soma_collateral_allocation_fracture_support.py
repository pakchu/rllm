"""Build outcome-blind SCAF-48 source-support and SLCS novelty evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import errno
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from training import preregister_soma_collateral_allocation_fracture as prereg


PROTOCOL_VERSION = "soma_collateral_allocation_fracture_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/build_soma_collateral_allocation_fracture_support.py"
)
TEST_PATH = Path(
    "tests/test_build_soma_collateral_allocation_fracture_support.py"
)
IMPLEMENTATION_CONTRACT = Path(
    "docs/scaf-source-support-implementation-contract-2026-07-24.md"
)
IMPLEMENTATION_CONTRACT_SHA256 = (
    "86a173fe14a8f7a8e1b9e7ef355186c1c636e0c101a48957f82f90fdfb76d890"
)
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "1542ed321e8fc64f49aeea6f2582db64f3cba31bfabf459425427874e37dfaca"
)
PREREGISTRATION_MANIFEST_HASH = (
    "fd22849e9a92476fc2e08805bd7e634cc478592238a087ef5a67c121d51f1a44"
)
PREREGISTRATION_BUILDER = prereg.SCRIPT_PATH
PREREGISTRATION_BUILDER_SHA256 = (
    "c4398128a031a7b3d1d9aeff8a3e07e5f7454f13d5cded3423ee70051dfc5fec"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "data/soma_collateral_allocation_fracture_clocks_2020_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/soma_collateral_allocation_fracture_support_2026-07-24.json"
)

UTC = ZoneInfo("UTC")
NEW_YORK = ZoneInfo("America/New_York")
BAR = pd.Timedelta(minutes=5)
HOLD = pd.Timedelta(minutes=prereg.Policy().hold_minutes)
COMMON_START = pd.Timestamp(prereg.Policy().train_start)
TRAIN_END = pd.Timestamp(prereg.Policy().train_end)
COMMON_END = pd.Timestamp(prereg.Policy().selection_end)
QUANTUM = Decimal("0.000000000001")
NONNEGATIVE_DECIMAL = re.compile(r"(?:0|[1-9]\d*)(?:\.\d+)?$")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}$")
RFC3339_UTC_SECONDS = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|\+00:00)$"
)

CLOCK_COLUMNS = (
    "control",
    "signal_id",
    "signal_available_time",
    "entry_time",
    "exit_time",
    "side",
    "relation",
    "inventory_mismatch_direction",
    "award_distortion_direction",
    "unmet_demand_mass_direction",
    "fee_distortion_direction",
    "agreement_count",
    "prior_relation",
)
FORBIDDEN_CLOCK_TOKENS = (
    "amount",
    "rate",
    "jsd",
    "rank",
    "price",
    "open",
    "high",
    "low",
    "close",
    "return",
    "future",
    "label",
    "funding",
    "pnl",
    "reward",
    "cagr",
    "mdd",
    "operation_id",
    "cusip",
)
SOURCE_CLOCK_CONTROLS = prereg.CONTROL_ORDER[:10]
PRIMARY_CLOCK_SIDE_CONTROLS = prereg.CONTROL_ORDER[10:]


@dataclass(frozen=True)
class BatchFeature:
    signal_time: pd.Timestamp
    operation_count: int
    atom_count: int
    valid: bool
    invalid_reason: str | None
    components: tuple[Decimal, Decimal, Decimal, Decimal] | None
    permuted_components: tuple[Decimal, Decimal, Decimal, Decimal] | None


@dataclass(frozen=True)
class Transition:
    signal_time: pd.Timestamp
    directions: tuple[int, int, int, int]
    relation: str
    side_sign: int
    prior_relation: str


class ComparatorContractFailure(RuntimeError):
    """Carry deterministic comparator failure evidence into the report."""

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


def canonical_time(value: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("SCAF-48 timestamp is timezone-naive")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.microsecond:
        raise RuntimeError("SCAF-48 timestamp has fractional seconds")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp(value: Any, label: str) -> pd.Timestamp:
    text = str(value)
    if not RFC3339_UTC_SECONDS.fullmatch(text):
        raise RuntimeError(f"SCAF-48 noncanonical UTC timestamp: {label}")
    try:
        parsed = pd.Timestamp(text)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"SCAF-48 invalid timestamp: {label}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(f"SCAF-48 timestamp is not UTC: {label}")
    parsed = parsed.tz_convert("UTC")
    if parsed.microsecond:
        raise RuntimeError(f"SCAF-48 timestamp has fractional seconds: {label}")
    return parsed


def _parse_date(value: Any, label: str) -> str:
    text = str(value)
    if not ISO_DATE.fullmatch(text):
        raise RuntimeError(f"SCAF-48 invalid ISO date: {label}")
    try:
        if pd.Timestamp(text).strftime("%Y-%m-%d") != text:
            raise ValueError(text)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"SCAF-48 invalid ISO date: {label}") from error
    return text


def _parse_decimal(
    value: Any,
    label: str,
    *,
    optional: bool = False,
) -> Decimal | None:
    text = str(value)
    if optional and text == "":
        return None
    if not NONNEGATIVE_DECIMAL.fullmatch(text):
        raise RuntimeError(f"SCAF-48 noncanonical decimal: {label}")
    try:
        parsed = Decimal(text)
    except InvalidOperation as error:
        raise RuntimeError(f"SCAF-48 invalid decimal: {label}") from error
    if not parsed.is_finite() or parsed < 0:
        raise RuntimeError(f"SCAF-48 invalid nonnegative decimal: {label}")
    return parsed


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
        PREREGISTRATION_BUILDER,
    )
    relative = [str(path) for path in paths]
    tracked = _git_check("ls-files", "--error-unmatch", "--", *relative)
    if tracked.returncode:
        raise RuntimeError("SCAF-48 protocol files are not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *relative)
    if clean.returncode:
        raise RuntimeError("SCAF-48 protocol files differ from HEAD")


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION_BUILDER) != PREREGISTRATION_BUILDER_SHA256:
        raise RuntimeError("SCAF-48 preregistration builder hash drift")
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("SCAF-48 preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_manifest(payload)
    if payload["manifest_hash"] != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("SCAF-48 preregistration manifest hash drift")
    return payload


def verify_pre_source_bindings(
    preregistration: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    del preregistration
    prereg.validate_frozen_dependencies()
    if sha256_file(IMPLEMENTATION_CONTRACT) != IMPLEMENTATION_CONTRACT_SHA256:
        raise RuntimeError("SCAF-48 implementation contract hash drift")
    bindings = {
        **{
            path: {"path": path, "sha256": expected}
            for path, expected in prereg.frozen_dependencies().items()
        },
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
        },
        "preregistration_builder": {
            "path": str(PREREGISTRATION_BUILDER),
            "sha256": PREREGISTRATION_BUILDER_SHA256,
        },
        "implementation_contract": {
            "path": str(IMPLEMENTATION_CONTRACT),
            "sha256": IMPLEMENTATION_CONTRACT_SHA256,
        },
    }
    return bindings


def _read_source(
    path: Path,
    allowlist: Sequence[str],
) -> pd.DataFrame:
    frame = pd.read_csv(
        _path(path),
        usecols=list(allowlist),
        dtype="string",
        keep_default_na=False,
        na_filter=False,
    )
    return frame.loc[:, list(allowlist)]


def validate_operations(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != list(prereg.OPERATIONS_ALLOWLIST):
        raise RuntimeError("SCAF-48 operation columns differ from allowlist")
    rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        operation_id = str(row["operation_id"])
        if not operation_id or "\x00" in operation_id:
            raise RuntimeError("SCAF-48 invalid operation id")
        rows.append(
            {
                "operation_id": operation_id,
                "operation_date": _parse_date(
                    row["operation_date"], f"operation[{index}] date"
                ),
                "available_at_utc": _parse_timestamp(
                    row["available_at_utc"],
                    f"operation[{index}] availability",
                ),
                "total_par_submitted": _parse_decimal(
                    row["total_par_submitted"],
                    f"operation[{index}] submitted",
                ),
                "total_par_accepted": _parse_decimal(
                    row["total_par_accepted"],
                    f"operation[{index}] accepted",
                ),
            }
        )
    validated = pd.DataFrame(rows)
    if validated.empty:
        raise RuntimeError("SCAF-48 operation source is empty")
    if validated["operation_id"].duplicated().any():
        raise RuntimeError("SCAF-48 operation ids are duplicated")
    return validated.sort_values(
        ["available_at_utc", "operation_id"], kind="mergesort"
    ).reset_index(drop=True)


def validate_details(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != list(prereg.DETAILS_ALLOWLIST):
        raise RuntimeError("SCAF-48 detail columns differ from allowlist")
    rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        operation_id = str(row["operation_id"])
        cusip = str(row["cusip"])
        if (
            not operation_id
            or not cusip
            or "\x00" in operation_id
            or "\x00" in cusip
        ):
            raise RuntimeError("SCAF-48 invalid detail identity")
        submitted = _parse_decimal(
            row["par_submitted"], f"detail[{index}] submitted"
        )
        accepted = _parse_decimal(
            row["par_accepted"], f"detail[{index}] accepted"
        )
        available = _parse_decimal(
            row["actual_available_to_borrow"],
            f"detail[{index}] actual available",
        )
        fee = _parse_decimal(
            row["weighted_average_rate"],
            f"detail[{index}] fee",
            optional=True,
        )
        assert submitted is not None
        assert accepted is not None
        assert available is not None
        if accepted > submitted:
            raise RuntimeError("SCAF-48 accepted exceeds submitted")
        if accepted > 0 and fee is None:
            raise RuntimeError("SCAF-48 positive award has no fee")
        rows.append(
            {
                "operation_id": operation_id,
                "operation_date": _parse_date(
                    row["operation_date"], f"detail[{index}] date"
                ),
                "available_at_utc": _parse_timestamp(
                    row["available_at_utc"],
                    f"detail[{index}] availability",
                ),
                "cusip": cusip,
                "par_submitted": submitted,
                "par_accepted": accepted,
                "weighted_average_rate": fee,
                "actual_available_to_borrow": available,
            }
        )
    validated = pd.DataFrame(rows)
    if validated.empty:
        raise RuntimeError("SCAF-48 detail source is empty")
    if validated.duplicated(["operation_id", "cusip"]).any():
        raise RuntimeError("SCAF-48 operation/CUSIP rows are duplicated")
    return validated.sort_values(
        ["available_at_utc", "operation_id", "cusip"], kind="mergesort"
    ).reset_index(drop=True)


def reconcile_source(
    operations: pd.DataFrame,
    details: pd.DataFrame,
) -> None:
    operation_lookup = {
        row.operation_id: row for row in operations.itertuples(index=False)
    }
    unknown = set(details["operation_id"]) - set(operation_lookup)
    if unknown:
        raise RuntimeError("SCAF-48 detail references unknown operation")
    missing = set(operation_lookup) - set(details["operation_id"])
    if missing:
        raise RuntimeError("SCAF-48 operation has no detail rows")
    for operation_id, group in details.groupby("operation_id", sort=False):
        operation = operation_lookup[str(operation_id)]
        if not group["operation_date"].eq(operation.operation_date).all():
            raise RuntimeError("SCAF-48 operation/detail date mismatch")
        if not group["available_at_utc"].eq(operation.available_at_utc).all():
            raise RuntimeError(
                "SCAF-48 operation/detail availability mismatch"
            )
        submitted = sum(group["par_submitted"], Decimal(0))
        accepted = sum(group["par_accepted"], Decimal(0))
        if submitted != operation.total_par_submitted:
            raise RuntimeError("SCAF-48 submitted total does not reconcile")
        if accepted != operation.total_par_accepted:
            raise RuntimeError("SCAF-48 accepted total does not reconcile")


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    operations = validate_operations(
        _read_source(prereg.OPERATIONS, prereg.OPERATIONS_ALLOWLIST)
    )
    details = validate_details(
        _read_source(prereg.DETAILS, prereg.DETAILS_ALLOWLIST)
    )
    reconcile_source(operations, details)
    return operations, details


def _binary_shares(values: Sequence[Decimal], total: Decimal) -> list[float]:
    with localcontext() as context:
        context.prec = 80
        return [float(value / total) for value in values]


def _quantize_bounded(value: float, label: str) -> Decimal:
    if not math.isfinite(value):
        raise RuntimeError(f"SCAF-48 nonfinite component: {label}")
    tolerance = 1e-12
    if value < -tolerance or value > 1.0 + tolerance:
        raise RuntimeError(f"SCAF-48 component outside [0,1]: {label}")
    clipped = min(1.0, max(0.0, value))
    return Decimal(str(clipped)).quantize(QUANTUM, rounding=ROUND_HALF_EVEN)


def jsd(left: Sequence[float], right: Sequence[float]) -> Decimal:
    if len(left) != len(right) or not left:
        raise RuntimeError("SCAF-48 invalid JSD vectors")
    terms: list[float] = []
    for first, second in zip(left, right, strict=True):
        if (
            not math.isfinite(first)
            or not math.isfinite(second)
            or first < 0
            or second < 0
        ):
            raise RuntimeError("SCAF-48 invalid JSD probability")
        midpoint = 0.5 * (first + second)
        if midpoint <= 0:
            continue
        if first > 0:
            terms.append(0.5 * first * math.log(first / midpoint))
        if second > 0:
            terms.append(0.5 * second * math.log(second / midpoint))
    return _quantize_bounded(
        math.fsum(terms) / math.log(2.0),
        "JSD",
    )


def _unmet_mass(
    submitted: Sequence[Decimal],
    accepted: Sequence[Decimal],
) -> Decimal:
    denominator = sum(submitted, Decimal(0))
    numerator = sum(
        (
            amount
            for amount, award in zip(submitted, accepted, strict=True)
            if award == 0
        ),
        Decimal(0),
    )
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_HALF_EVEN
        return (numerator / denominator).quantize(
            QUANTUM,
            rounding=ROUND_HALF_EVEN,
        )


def _permutation_destinations(
    atoms: Sequence[tuple[str, str]],
    signal_time: pd.Timestamp,
) -> list[int]:
    timestamp = canonical_time(signal_time).encode("ascii")
    keyed: list[tuple[bytes, str, str, int]] = []
    for index, (operation_id, cusip) in enumerate(atoms):
        if "\x00" in operation_id or "\x00" in cusip:
            raise RuntimeError("SCAF-48 NUL in permutation identity")
        digest = hashlib.sha256(
            b"SCAF-48\x00"
            + timestamp
            + b"\x00"
            + operation_id.encode("utf-8")
            + b"\x00"
            + cusip.encode("utf-8")
        ).digest()
        keyed.append((digest, operation_id, cusip, index))
    return [item[3] for item in sorted(keyed)]


def _component_tuple(
    submitted: Sequence[Decimal],
    available: Sequence[Decimal],
    accepted: Sequence[Decimal],
    fees: Sequence[Decimal],
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    total_submitted = sum(submitted, Decimal(0))
    total_available = sum(available, Decimal(0))
    total_accepted = sum(accepted, Decimal(0))
    fee_mass = [
        award * fee
        for award, fee in zip(accepted, fees, strict=True)
    ]
    total_fee_mass = sum(fee_mass, Decimal(0))
    if min(
        total_submitted,
        total_available,
        total_accepted,
        total_fee_mass,
    ) <= 0:
        raise RuntimeError("SCAF-48 nonpositive required batch total")
    submitted_share = _binary_shares(submitted, total_submitted)
    available_share = _binary_shares(available, total_available)
    accepted_share = _binary_shares(accepted, total_accepted)
    fee_share = _binary_shares(fee_mass, total_fee_mass)
    return (
        jsd(submitted_share, available_share),
        jsd(submitted_share, accepted_share),
        _unmet_mass(submitted, accepted),
        jsd(accepted_share, fee_share),
    )


def build_batch_features(
    operations: pd.DataFrame,
    details: pd.DataFrame,
) -> list[BatchFeature]:
    reconcile_source(operations, details)
    operation_counts = (
        operations.groupby("available_at_utc", sort=True)["operation_id"]
        .count()
        .to_dict()
    )
    features: list[BatchFeature] = []
    for signal_time, group in details.groupby("available_at_utc", sort=True):
        ordered = group.sort_values(
            ["operation_id", "cusip"], kind="mergesort"
        )
        atoms = [
            (str(row.operation_id), str(row.cusip))
            for row in ordered.itertuples(index=False)
        ]
        submitted = list(ordered["par_submitted"])
        available = list(ordered["actual_available_to_borrow"])
        accepted = list(ordered["par_accepted"])
        fees = [
            Decimal(0) if value is None else value
            for value in ordered["weighted_average_rate"]
        ]
        timestamp = pd.Timestamp(signal_time)
        try:
            components = _component_tuple(
                submitted,
                available,
                accepted,
                fees,
            )
            destinations = _permutation_destinations(atoms, timestamp)
            permuted_submitted = [Decimal(0)] * len(submitted)
            for source_index, destination_index in enumerate(destinations):
                permuted_submitted[destination_index] = submitted[source_index]
            permuted = _component_tuple(
                permuted_submitted,
                available,
                accepted,
                fees,
            )
            valid = True
            reason = None
        except RuntimeError as error:
            components = None
            permuted = None
            valid = False
            reason = str(error)
        features.append(
            BatchFeature(
                signal_time=timestamp,
                operation_count=int(operation_counts[timestamp]),
                atom_count=len(ordered),
                valid=valid,
                invalid_reason=reason,
                components=components,
                permuted_components=permuted,
            )
        )
    if not features:
        raise RuntimeError("SCAF-48 produced no causal batches")
    return features


def _directions(
    current: Sequence[Decimal],
    previous: Sequence[Decimal],
) -> tuple[int, int, int, int]:
    values = tuple(
        1 if now > prior else -1 if now < prior else 0
        for now, prior in zip(current, previous, strict=True)
    )
    if len(values) != 4:
        raise RuntimeError("SCAF-48 direction vector length drift")
    return values  # type: ignore[return-value]


def _relation(directions: Sequence[int]) -> tuple[str, int]:
    up = sum(value == 1 for value in directions)
    down = sum(value == -1 for value in directions)
    if up >= 3:
        return "FRACTURE", -1
    if down >= 3:
        return "RELIEF", 1
    return "NEUTRAL", 0


def _direction_token(value: int) -> str:
    return {1: "UP", -1: "DOWN", 0: "FLAT"}[int(value)]


def _primary_signal_id(signal_time: pd.Timestamp, relation: str) -> str:
    payload = f"SCAF-48|{canonical_time(signal_time)}|{relation}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _control_signal_id(
    control: str,
    signal_time: pd.Timestamp,
    side_sign: int,
) -> str:
    side = "LONG" if side_sign == 1 else "SHORT"
    payload = (
        f"SCAF-48|{control}|{canonical_time(signal_time)}|{side}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate(
    *,
    control: str,
    signal_time: pd.Timestamp,
    relation: str,
    side_sign: int,
    directions: tuple[int, int, int, int],
    prior_relation: str,
) -> dict[str, Any]:
    if side_sign not in (-1, 1):
        raise RuntimeError("SCAF-48 candidate side is invalid")
    signal_id = (
        _primary_signal_id(signal_time, relation)
        if control == "primary"
        else _control_signal_id(control, signal_time, side_sign)
    )
    return {
        "control": control,
        "signal_id": signal_id,
        "signal_available_time": signal_time,
        "relation": relation,
        "side_sign": side_sign,
        "directions": directions,
        "agreement_count": max(
            sum(value == 1 for value in directions),
            sum(value == -1 for value in directions),
        ),
        "prior_relation": prior_relation,
    }


def _mean_component(values: Sequence[Decimal]) -> Decimal:
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_HALF_EVEN
        return (sum(values, Decimal(0)) / Decimal(4)).quantize(
            QUANTUM,
            rounding=ROUND_HALF_EVEN,
        )


def build_raw_candidates(
    features: Sequence[BatchFeature],
) -> tuple[dict[str, list[dict[str, Any]]], list[Transition]]:
    raw = {control: [] for control in SOURCE_CLOCK_CONTROLS}
    transitions: list[Transition] = []
    previous: BatchFeature | None = None
    prior_relation = "BASELINE"
    transition_history: list[tuple[int, int, int, int]] = []
    for feature in sorted(features, key=lambda item: item.signal_time):
        if not feature.valid:
            previous = None
            prior_relation = "RESET"
            transition_history.clear()
            continue
        if previous is None:
            previous = feature
            prior_relation = "BASELINE"
            transition_history.clear()
            continue
        assert feature.components is not None
        assert feature.permuted_components is not None
        assert previous.components is not None
        assert previous.permuted_components is not None
        directions = _directions(feature.components, previous.components)
        relation, side = _relation(directions)
        transitions.append(
            Transition(
                signal_time=feature.signal_time,
                directions=directions,
                relation=relation,
                side_sign=side,
                prior_relation=prior_relation,
            )
        )
        if side:
            raw["primary"].append(
                _candidate(
                    control="primary",
                    signal_time=feature.signal_time,
                    relation=relation,
                    side_sign=side,
                    directions=directions,
                    prior_relation=prior_relation,
                )
            )
        for index, control in enumerate(SOURCE_CLOCK_CONTROLS[1:5]):
            component_side = -directions[index]
            if component_side:
                raw[control].append(
                    _candidate(
                        control=control,
                        signal_time=feature.signal_time,
                        relation=(
                            f"{prereg.COMPONENT_ORDER[index].upper()}_"
                            f"{_direction_token(directions[index])}"
                        ),
                        side_sign=component_side,
                        directions=directions,
                        prior_relation=prior_relation,
                    )
                )
        mean_direction = (
            1
            if _mean_component(feature.components)
            > _mean_component(previous.components)
            else -1
            if _mean_component(feature.components)
            < _mean_component(previous.components)
            else 0
        )
        if mean_direction:
            raw["mean_change_without_consensus"].append(
                _candidate(
                    control="mean_change_without_consensus",
                    signal_time=feature.signal_time,
                    relation=f"MEAN_{_direction_token(mean_direction)}",
                    side_sign=-mean_direction,
                    directions=directions,
                    prior_relation=prior_relation,
                )
            )
        up = sum(value == 1 for value in directions)
        down = sum(value == -1 for value in directions)
        if up >= 2 and down == 0:
            two_side = -1
            two_relation = "TWO_UP_NO_DOWN"
        elif down >= 2 and up == 0:
            two_side = 1
            two_relation = "TWO_DOWN_NO_UP"
        else:
            two_side = 0
            two_relation = "NEUTRAL"
        if two_side:
            raw["two_of_four_without_opposition"].append(
                _candidate(
                    control="two_of_four_without_opposition",
                    signal_time=feature.signal_time,
                    relation=two_relation,
                    side_sign=two_side,
                    directions=directions,
                    prior_relation=prior_relation,
                )
            )
        for lag, control in (
            (1, "one_batch_stale"),
            (5, "five_batch_stale"),
        ):
            if len(transition_history) >= lag:
                stale_directions = transition_history[-lag]
                stale_relation, stale_side = _relation(stale_directions)
                if stale_side:
                    raw[control].append(
                        _candidate(
                            control=control,
                            signal_time=feature.signal_time,
                            relation=f"STALE_{lag}_{stale_relation}",
                            side_sign=stale_side,
                            directions=stale_directions,
                            prior_relation=prior_relation,
                        )
                    )
        permuted_directions = _directions(
            feature.permuted_components,
            previous.permuted_components,
        )
        permuted_relation, permuted_side = _relation(permuted_directions)
        if permuted_side:
            raw["within_batch_demand_permutation"].append(
                _candidate(
                    control="within_batch_demand_permutation",
                    signal_time=feature.signal_time,
                    relation=f"PERMUTED_{permuted_relation}",
                    side_sign=permuted_side,
                    directions=permuted_directions,
                    prior_relation=prior_relation,
                )
            )
        transition_history.append(directions)
        previous = feature
        prior_relation = relation
    return raw, transitions


def _ceil_to_5m(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.floor("5min") == timestamp:
        return timestamp
    return timestamp.ceil("5min")


def _split(entry: pd.Timestamp, exit_time: pd.Timestamp) -> str | None:
    if COMMON_START <= entry and exit_time <= TRAIN_END:
        return "train"
    if TRAIN_END <= entry and exit_time <= COMMON_END:
        return "selection"
    return None


def _schedule(
    control: str,
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for candidate in candidates:
        signal = pd.Timestamp(candidate["signal_available_time"])
        entry = _ceil_to_5m(signal) + BAR
        exit_time = entry + HOLD
        prepared.append(
            {
                **dict(candidate),
                "control": control,
                "entry_time": entry,
                "exit_time": exit_time,
            }
        )
    ordered = sorted(
        prepared,
        key=lambda row: (
            row["entry_time"],
            row["signal_available_time"],
            row["signal_id"],
        ),
    )
    accepted: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    for row in ordered:
        if previous_exit is not None and row["entry_time"] < previous_exit:
            continue
        previous_exit = row["exit_time"]
        split = _split(row["entry_time"], row["exit_time"])
        if split is None:
            continue
        accepted.append({**row, "split": split})
    return accepted


def _random_side(primary_signal_id: str) -> int:
    payload = (
        f"SCAF-48|{primary_signal_id}|RANDOM_SIDE".encode("utf-8")
    )
    number = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    return 1 if number % 2 == 0 else -1


def build_clocks(
    raw: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    clocks = {
        control: _schedule(control, raw[control])
        for control in SOURCE_CLOCK_CONTROLS
    }
    primary = clocks["primary"]
    for control in PRIMARY_CLOCK_SIDE_CONTROLS:
        rows: list[dict[str, Any]] = []
        for row in primary:
            if control == "exact_direction_flip":
                side = -int(row["side_sign"])
            elif control == "deterministic_random_side":
                side = _random_side(str(row["signal_id"]))
            elif control == "constant_long":
                side = 1
            elif control == "constant_short":
                side = -1
            else:
                raise RuntimeError(f"SCAF-48 unknown side control: {control}")
            rows.append(
                {
                    **row,
                    "control": control,
                    "side_sign": side,
                    "relation": control.upper(),
                    "signal_id": _control_signal_id(
                        control,
                        pd.Timestamp(row["signal_available_time"]),
                        side,
                    ),
                }
            )
        clocks[control] = rows
    if tuple(clocks) != prereg.CONTROL_ORDER:
        raise RuntimeError("SCAF-48 control order drift")
    return clocks


def _hypothetical_split(signal_time: pd.Timestamp) -> str | None:
    entry = _ceil_to_5m(signal_time) + BAR
    return _split(entry, entry + HOLD)


def _rows(
    clocks: Mapping[str, Sequence[Mapping[str, Any]]],
    control: str,
    split: str,
) -> list[Mapping[str, Any]]:
    return [
        row for row in clocks[control] if str(row["split"]) == split
    ]


def clock_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: pd.Timestamp(row["entry_time"]))
    events = len(ordered)
    sides = [int(row["side_sign"]) for row in ordered]
    entries = [pd.Timestamp(row["entry_time"]).tz_convert("UTC") for row in ordered]
    months = [entry.strftime("%Y-%m") for entry in entries]
    quarters = [
        f"{entry.year}-Q{((entry.month - 1) // 3) + 1}"
        for entry in entries
    ]
    dates = [entry.date() for entry in entries]
    gaps = [
        (current - previous).days
        for previous, current in zip(dates, dates[1:])
    ]
    maximum_run = 0
    current_run = 0
    previous_side: int | None = None
    for side in sides:
        current_run = current_run + 1 if side == previous_side else 1
        maximum_run = max(maximum_run, current_run)
        previous_side = side
    month_counts = Counter(months)
    quarter_counts = Counter(quarters)
    year_counts = Counter(entry.year for entry in entries)
    half_counts = Counter(
        f"{entry.year}-H{1 if entry.month <= 6 else 2}" for entry in entries
    )
    return {
        "events": events,
        "long": sum(side == 1 for side in sides),
        "short": sum(side == -1 for side in sides),
        "active_months": len(month_counts),
        "maximum_gap_days": max(gaps) if gaps else None,
        "maximum_month_share": (
            max(month_counts.values()) / events if events else None
        ),
        "maximum_quarter_share": (
            max(quarter_counts.values()) / events if events else None
        ),
        "maximum_same_side_run": maximum_run,
        "year_counts": {
            str(year): year_counts.get(year, 0)
            for year in range(2020, 2024)
        },
        "half_counts": {
            f"2023-H{half}": half_counts.get(f"2023-H{half}", 0)
            for half in (1, 2)
        },
        "quarter_counts": {
            f"2023-Q{quarter}": quarter_counts.get(
                f"2023-Q{quarter}", 0
            )
            for quarter in range(1, 5)
        },
    }


def _same_side_reproduction(
    primary: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
) -> float | None:
    if not primary:
        return None
    lookup = {
        (pd.Timestamp(row["entry_time"]), int(row["side_sign"]))
        for row in control
    }
    matched = sum(
        (pd.Timestamp(row["entry_time"]), int(row["side_sign"])) in lookup
        for row in primary
    )
    return matched / len(primary)


def _exact_entry_jaccard(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> float | None:
    left = {pd.Timestamp(row["entry_time"]) for row in first}
    right = {pd.Timestamp(row["entry_time"]) for row in second}
    union = left | right
    return len(left & right) / len(union) if union else None


def support_and_composition(
    features: Sequence[BatchFeature],
    transitions: Sequence[Transition],
    raw: Mapping[str, Sequence[Mapping[str, Any]]],
    clocks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[
    dict[str, Any],
    dict[str, bool],
    dict[str, Any],
    dict[str, bool],
]:
    complete_counts = {
        split: sum(
            feature.valid
            and (
                (COMMON_START <= feature.signal_time < TRAIN_END)
                if split == "train"
                else (TRAIN_END <= feature.signal_time < COMMON_END)
            )
            for feature in features
        )
        for split in ("train", "selection")
    }
    transition_counts = {
        split: sum(
            (
                (COMMON_START <= row.signal_time < TRAIN_END)
                if split == "train"
                else (TRAIN_END <= row.signal_time < COMMON_END)
            )
            for row in transitions
        )
        for split in ("train", "selection")
    }
    eligible_transition_counts = {
        split: sum(
            _hypothetical_split(row.signal_time) == split
            for row in transitions
        )
        for split in ("train", "selection")
    }
    raw_primary = {
        split: [
            row
            for row in raw["primary"]
            if _hypothetical_split(
                pd.Timestamp(row["signal_available_time"])
            )
            == split
        ]
        for split in ("train", "selection")
    }
    raw_consensus_share = {
        split: (
            len(raw_primary[split]) / eligible_transition_counts[split]
            if eligible_transition_counts[split]
            else None
        )
        for split in ("train", "selection")
    }
    primary_stats = {
        split: clock_stats(_rows(clocks, "primary", split))
        for split in ("train", "selection")
    }
    coverage_gate = prereg.build_manifest()["source_support_gate"]["coverage"]
    train_gate = prereg.build_manifest()["source_support_gate"]["train"]
    selection_gate = prereg.build_manifest()["source_support_gate"]["selection"]
    source_checks: dict[str, bool] = {
        "train_complete_batches_min": (
            complete_counts["train"]
            >= coverage_gate["train_complete_batches_min"]
        ),
        "selection_complete_batches_min": (
            complete_counts["selection"]
            >= coverage_gate["selection_complete_batches_min"]
        ),
        "train_valid_transitions_min": (
            transition_counts["train"]
            >= coverage_gate["train_valid_transitions_min"]
        ),
        "selection_valid_transitions_min": (
            transition_counts["selection"]
            >= coverage_gate["selection_valid_transitions_min"]
        ),
    }
    for split in ("train", "selection"):
        share = raw_consensus_share[split]
        source_checks[f"{split}_raw_consensus_share_min"] = bool(
            share is not None
            and share
            >= coverage_gate["each_split_raw_consensus_share_min"]
        )
        source_checks[f"{split}_raw_consensus_share_max"] = bool(
            share is not None
            and share
            <= coverage_gate["each_split_raw_consensus_share_max"]
        )
    train = primary_stats["train"]
    source_checks.update(
        {
            "train_events_min": train["events"] >= train_gate["events_min"],
            "train_each_year_events_min": all(
                train["year_counts"][str(year)]
                >= train_gate["each_year_events_min"]
                for year in (2020, 2021, 2022)
            ),
            "train_long_min": train["long"] >= train_gate["long_min"],
            "train_short_min": train["short"] >= train_gate["short_min"],
            "train_active_months_min": (
                train["active_months"] >= train_gate["active_months_min"]
            ),
            "train_maximum_gap": bool(
                train["maximum_gap_days"] is not None
                and train["maximum_gap_days"]
                <= train_gate["maximum_utc_calendar_gap_days"]
            ),
            "train_maximum_month_share": bool(
                train["maximum_month_share"] is not None
                and train["maximum_month_share"]
                <= train_gate["maximum_month_share"]
            ),
            "train_maximum_quarter_share": bool(
                train["maximum_quarter_share"] is not None
                and train["maximum_quarter_share"]
                <= train_gate["maximum_quarter_share"]
            ),
            "train_maximum_same_side_run": (
                train["maximum_same_side_run"]
                <= train_gate["maximum_same_side_run"]
            ),
        }
    )
    selection = primary_stats["selection"]
    source_checks.update(
        {
            "selection_events_min": (
                selection["events"] >= selection_gate["events_min"]
            ),
            "selection_each_half_events_min": all(
                selection["half_counts"][f"2023-H{half}"]
                >= selection_gate["each_half_events_min"]
                for half in (1, 2)
            ),
            "selection_each_quarter_events_min": all(
                selection["quarter_counts"][f"2023-Q{quarter}"]
                >= selection_gate["each_quarter_events_min"]
                for quarter in range(1, 5)
            ),
            "selection_long_min": (
                selection["long"] >= selection_gate["long_min"]
            ),
            "selection_short_min": (
                selection["short"] >= selection_gate["short_min"]
            ),
            "selection_active_months_min": (
                selection["active_months"]
                >= selection_gate["active_months_min"]
            ),
            "selection_maximum_gap": bool(
                selection["maximum_gap_days"] is not None
                and selection["maximum_gap_days"]
                <= selection_gate["maximum_utc_calendar_gap_days"]
            ),
            "selection_maximum_month_share": bool(
                selection["maximum_month_share"] is not None
                and selection["maximum_month_share"]
                <= selection_gate["maximum_month_share"]
            ),
            "selection_maximum_same_side_run": (
                selection["maximum_same_side_run"]
                <= selection_gate["maximum_same_side_run"]
            ),
        }
    )
    for split in ("train", "selection"):
        for control in prereg.CONTROL_ORDER:
            source_checks[f"{split}:required_control:{control}"] = bool(
                _rows(clocks, control, split)
            )
    composition: dict[str, Any] = {}
    composition_checks: dict[str, bool] = {}
    composition_gate = prereg.build_manifest()["composition_gate"]["each_split"]
    for split in ("train", "selection"):
        raw_rows = raw_primary[split]
        denominator = len(raw_rows)
        agreement: dict[str, float | None] = {}
        for index, component in enumerate(prereg.COMPONENT_ORDER):
            matched = sum(
                (
                    int(row["directions"][index]) == 1
                    and str(row["relation"]) == "FRACTURE"
                )
                or (
                    int(row["directions"][index]) == -1
                    and str(row["relation"]) == "RELIEF"
                )
                for row in raw_rows
            )
            agreement[component] = matched / denominator if denominator else None
        four = sum(int(row["agreement_count"]) == 4 for row in raw_rows)
        three = sum(int(row["agreement_count"]) == 3 for row in raw_rows)
        metrics: dict[str, Any] = {
            "raw_primary_opportunities": denominator,
            "component_raw_agreement": agreement,
            "four_of_four_share": four / denominator if denominator else None,
            "exact_three_of_four_share": (
                three / denominator if denominator else None
            ),
        }
        primary_rows = _rows(clocks, "primary", split)
        reproduction_controls = {
            "inventory_mismatch_only": "each_component_control_reproduction_max",
            "award_distortion_only": "each_component_control_reproduction_max",
            "unmet_demand_mass_only": "each_component_control_reproduction_max",
            "fee_distortion_only": "each_component_control_reproduction_max",
            "mean_change_without_consensus": "mean_change_reproduction_max",
            "one_batch_stale": "each_stale_reproduction_max",
            "five_batch_stale": "each_stale_reproduction_max",
            "deterministic_random_side": "random_side_reproduction_max",
        }
        metrics["control_reproduction"] = {
            control: _same_side_reproduction(
                primary_rows,
                _rows(clocks, control, split),
            )
            for control in reproduction_controls
        }
        permutation = _rows(
            clocks, "within_batch_demand_permutation", split
        )
        metrics["permutation_exact_entry_jaccard"] = _exact_entry_jaccard(
            primary_rows, permutation
        )
        metrics["permutation_same_side_reproduction"] = (
            _same_side_reproduction(primary_rows, permutation)
        )
        composition[split] = metrics
        for component, value in agreement.items():
            composition_checks[f"{split}:{component}:agreement_min"] = bool(
                value is not None
                and value
                >= composition_gate["each_component_raw_agreement_min"]
            )
            composition_checks[f"{split}:{component}:agreement_max"] = bool(
                value is not None
                and value
                <= composition_gate["each_component_raw_agreement_max"]
            )
        composition_checks[f"{split}:four_of_four_share_min"] = bool(
            metrics["four_of_four_share"] is not None
            and metrics["four_of_four_share"]
            >= composition_gate["four_of_four_share_min"]
        )
        composition_checks[f"{split}:four_of_four_share_max"] = bool(
            metrics["four_of_four_share"] is not None
            and metrics["four_of_four_share"]
            <= composition_gate["four_of_four_share_max"]
        )
        composition_checks[f"{split}:exact_three_of_four_share_min"] = bool(
            metrics["exact_three_of_four_share"] is not None
            and metrics["exact_three_of_four_share"]
            >= composition_gate["exact_three_of_four_share_min"]
        )
        for control, gate_name in reproduction_controls.items():
            value = metrics["control_reproduction"][control]
            composition_checks[f"{split}:{control}:reproduction"] = bool(
                value is not None and value <= composition_gate[gate_name]
            )
        entry_jaccard = metrics["permutation_exact_entry_jaccard"]
        composition_checks[f"{split}:permutation_entry_jaccard"] = bool(
            entry_jaccard is not None
            and entry_jaccard
            <= composition_gate["permutation_exact_entry_jaccard_max"]
        )
        same_side = metrics["permutation_same_side_reproduction"]
        composition_checks[f"{split}:permutation_same_side"] = bool(
            same_side is not None
            and same_side
            <= composition_gate["permutation_same_side_reproduction_max"]
        )
    support_report = {
        "complete_batches": complete_counts,
        "valid_transitions": transition_counts,
        "split_contained_transition_denominators": eligible_transition_counts,
        "raw_primary_opportunities": {
            split: len(rows) for split, rows in raw_primary.items()
        },
        "raw_consensus_share": raw_consensus_share,
        "primary_clock": primary_stats,
        "invalid_batches": sum(not feature.valid for feature in features),
        "continuity_resets": sum(not feature.valid for feature in features),
    }
    return support_report, source_checks, composition, composition_checks


def first_failure(
    source_checks: Mapping[str, bool],
    composition_checks: Mapping[str, bool],
    novelty_checks: Mapping[str, bool],
    *,
    artifact_eligible: bool,
) -> tuple[str, str | None]:
    for name, passed in source_checks.items():
        if not passed:
            return "source_support", name
    for name, passed in composition_checks.items():
        if not passed:
            return "relational_composition", name
    if not artifact_eligible:
        return "artifact_eligibility", "synthetic_or_injected_build"
    for name, passed in novelty_checks.items():
        if not passed:
            return "comparator_novelty", name
    if not novelty_checks:
        return "comparator_novelty", "required_comparator_checks_missing"
    return "none", None


def _validate_interval_group(
    rows: pd.DataFrame,
    key: str,
) -> pd.DataFrame:
    ordered = rows.sort_values(
        ["entry_time", "original_row_number"], kind="mergesort"
    ).reset_index(drop=True)
    if ordered["entry_time"].duplicated().any():
        raise RuntimeError(f"SCAF-48 comparator duplicate entry: {key}")
    if not ordered["exit_time"].gt(ordered["entry_time"]).all():
        raise RuntimeError(f"SCAF-48 comparator invalid interval: {key}")
    if len(ordered) > 1:
        entries = ordered["entry_time"].iloc[1:].reset_index(drop=True)
        exits = ordered["exit_time"].iloc[:-1].reset_index(drop=True)
        if not entries.ge(exits).all():
            raise RuntimeError(f"SCAF-48 comparator self-overlap: {key}")
    return ordered


def _read_comparator_groups_impl(
    preregistration: Mapping[str, Any],
    decoded_counter: list[int],
) -> dict[str, pd.DataFrame]:
    contract = preregistration["novelty_contract"]["comparator"]
    path = Path(contract["path"])
    if sha256_file(path) != contract["sha256"]:
        raise RuntimeError("SCAF-48 comparator hash drift")
    if prereg.sha256_csv_header(path) != contract["header_sha256"]:
        raise RuntimeError("SCAF-48 comparator header hash drift")
    raw = pd.read_csv(
        _path(path),
        usecols=list(contract["read_csv"]["usecols"]),
        dtype="string",
        keep_default_na=False,
        na_filter=False,
    )
    raw = raw.loc[:, list(contract["read_csv"]["usecols"])].copy()
    decoded_counter[0] = len(raw)
    raw["original_row_number"] = np.arange(len(raw), dtype=np.int64)
    raw["entry_time"] = [
        _parse_timestamp(value, f"comparator[{index}] entry")
        for index, value in enumerate(raw["entry_time"])
    ]
    raw["exit_time"] = [
        _parse_timestamp(value, f"comparator[{index}] exit")
        for index, value in enumerate(raw["exit_time"])
    ]
    if not raw["side"].isin(("LONG", "SHORT")).all():
        raise RuntimeError("SCAF-48 comparator side invalid before filtering")
    if not raw["exit_time"].gt(raw["entry_time"]).all():
        raise RuntimeError(
            "SCAF-48 comparator interval invalid before filtering"
        )
    groups: dict[str, pd.DataFrame] = {}
    minimum = int(contract["minimum_contained_rows_each"])
    for group in contract["groups"]:
        selected = raw.loc[raw["control"].eq(group)].copy()
        if selected.empty:
            raise RuntimeError(f"SCAF-48 comparator group empty: {group}")
        selected = _validate_interval_group(selected, str(group))
        before = selected["exit_time"].le(COMMON_START)
        after = selected["entry_time"].ge(COMMON_END)
        contained_mask = selected["entry_time"].ge(
            COMMON_START
        ) & selected["exit_time"].le(COMMON_END)
        crossing = ~(before | after | contained_mask)
        contained = selected.loc[contained_mask].reset_index(drop=True)
        if len(contained) < minimum:
            raise RuntimeError(
                f"SCAF-48 comparator below contained floor: {group}"
            )
        contained.attrs["counts"] = {
            "raw_selected_rows": len(selected),
            "fully_contained_rows": len(contained),
            "before_window_rows": int(before.sum()),
            "after_window_rows": int(after.sum()),
            "boundary_crossing_rows": int(crossing.sum()),
        }
        groups[str(group)] = contained
    return groups


def _read_comparator_groups(
    preregistration: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], int]:
    counter = [0]
    try:
        return _read_comparator_groups_impl(preregistration, counter), counter[0]
    except ComparatorContractFailure:
        raise
    except Exception as error:
        raise ComparatorContractFailure(
            "comparator_artifact_contract",
            counter[0],
            str(error),
        ) from error


def _candidate_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entry_time": pd.Timestamp(row["entry_time"]),
                "exit_time": pd.Timestamp(row["exit_time"]),
                "side": "LONG" if int(row["side_sign"]) == 1 else "SHORT",
                "signal_id": str(row["signal_id"]),
                "original_row_number": index,
            }
            for index, row in enumerate(rows)
        ]
    )


def _exact_jaccard(left: pd.DataFrame, right: pd.DataFrame) -> float:
    first = set(left["entry_time"])
    second = set(right["entry_time"])
    denominator = len(first) + len(second) - len(first & second)
    if not denominator:
        raise RuntimeError("SCAF-48 novelty exact Jaccard empty")
    return len(first & second) / denominator


def _maximum_bipartite_matches(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> int:
    left_ordered = left.sort_values(
        ["entry_time", "signal_id"], kind="mergesort"
    ).reset_index(drop=True)
    right_ordered = right.sort_values(
        ["entry_time", "original_row_number"], kind="mergesort"
    ).reset_index(drop=True)
    right_dates = [
        timestamp.tz_convert(NEW_YORK).date()
        for timestamp in right_ordered["entry_time"]
    ]
    adjacency: list[list[int]] = []
    for timestamp in left_ordered["entry_time"]:
        date = timestamp.tz_convert(NEW_YORK).date()
        adjacency.append(
            [
                index
                for index, other in enumerate(right_dates)
                if abs((date - other).days) <= 1
            ]
        )
    match_right = [-1] * len(right_ordered)

    def augment(left_index: int, seen: set[int]) -> bool:
        for right_index in adjacency[left_index]:
            if right_index in seen:
                continue
            seen.add(right_index)
            prior = match_right[right_index]
            if prior == -1 or augment(prior, seen):
                match_right[right_index] = left_index
                return True
        return False

    matched = 0
    for left_index in range(len(left_ordered)):
        matched += int(augment(left_index, set()))
    return matched


def _one_day_jaccard(left: pd.DataFrame, right: pd.DataFrame) -> float:
    matched = _maximum_bipartite_matches(left, right)
    denominator = len(left) + len(right) - matched
    if not denominator:
        raise RuntimeError("SCAF-48 novelty one-day Jaccard empty")
    return matched / denominator


def _same_entry_same_side_reproduction(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
) -> float:
    if candidate.empty:
        raise RuntimeError("SCAF-48 novelty candidate denominator empty")
    lookup = set(zip(comparator["entry_time"], comparator["side"]))
    matched = sum(
        (row.entry_time, row.side) in lookup
        for row in candidate.itertuples(index=False)
    )
    return matched / len(candidate)


def _signed_occupancy(rows: pd.DataFrame) -> np.ndarray:
    size = int((COMMON_END - COMMON_START) / BAR)
    occupancy = np.zeros(size, dtype=np.int8)
    for row in rows.sort_values("entry_time", kind="mergesort").itertuples(
        index=False
    ):
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if (entry - COMMON_START) % BAR or (exit_time - COMMON_START) % BAR:
            raise RuntimeError("SCAF-48 novelty interval off five-minute grid")
        left = int((entry - COMMON_START) / BAR)
        right = int((exit_time - COMMON_START) / BAR)
        if left < 0 or right > size or left >= right:
            raise RuntimeError("SCAF-48 novelty interval outside common window")
        if np.any(occupancy[left:right] != 0):
            raise RuntimeError("SCAF-48 novelty clock self-overlap")
        occupancy[left:right] = 1 if row.side == "LONG" else -1
    return occupancy


def _occupancy_correlation(left: pd.DataFrame, right: pd.DataFrame) -> float:
    first = _signed_occupancy(left)
    second = _signed_occupancy(right)
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        raise RuntimeError("SCAF-48 novelty occupancy is constant")
    result = float(np.corrcoef(first, second)[0, 1])
    if not math.isfinite(result):
        raise RuntimeError("SCAF-48 novelty occupancy correlation nonfinite")
    return abs(result)


def evaluate_novelty(
    primary: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool], int]:
    counter = [0]
    try:
        candidate = _candidate_frame(primary)
        if candidate.empty:
            raise RuntimeError("SCAF-48 novelty primary is empty")
        _signed_occupancy(candidate)
        groups, decoded = _read_comparator_groups(preregistration)
        counter[0] = decoded
        thresholds = preregistration["novelty_contract"][
            "thresholds_each_group"
        ]
        report: dict[str, Any] = {}
        checks: dict[str, bool] = {}
        for name, comparator in groups.items():
            exact = _exact_jaccard(candidate, comparator)
            one_day = _one_day_jaccard(candidate, comparator)
            reproduction = _same_entry_same_side_reproduction(
                candidate, comparator
            )
            correlation = _occupancy_correlation(candidate, comparator)
            report[name] = {
                "candidate_rows": len(candidate),
                "comparator_rows": len(comparator),
                "comparator_counts": comparator.attrs["counts"],
                "exact_entry_jaccard": exact,
                "one_new_york_calendar_day_jaccard": one_day,
                "same_entry_same_side_reproduction": reproduction,
                "absolute_signed_occupancy_pearson": correlation,
            }
            checks[f"{name}:exact_entry_jaccard"] = (
                exact <= thresholds["exact_entry_jaccard_max"]
            )
            checks[f"{name}:one_calendar_day_jaccard"] = (
                one_day
                <= thresholds["one_new_york_calendar_day_jaccard_max"]
            )
            checks[f"{name}:same_entry_same_side_reproduction"] = (
                reproduction
                <= thresholds["same_entry_same_side_reproduction_max"]
            )
            checks[f"{name}:absolute_signed_occupancy_pearson"] = (
                correlation
                <= thresholds["absolute_signed_occupancy_pearson_max"]
            )
        return report, checks, decoded
    except ComparatorContractFailure:
        raise
    except Exception as error:
        raise ComparatorContractFailure(
            "novelty_metric_contract",
            counter[0],
            str(error),
        ) from error


def deterministic_clock_bytes(
    clocks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bytes:
    rows: list[dict[str, Any]] = []
    for control in prereg.CONTROL_ORDER:
        for row in clocks[control]:
            directions = tuple(int(value) for value in row["directions"])
            rows.append(
                {
                    "control": control,
                    "signal_id": str(row["signal_id"]),
                    "signal_available_time": canonical_time(
                        pd.Timestamp(row["signal_available_time"])
                    ),
                    "entry_time": canonical_time(
                        pd.Timestamp(row["entry_time"])
                    ),
                    "exit_time": canonical_time(
                        pd.Timestamp(row["exit_time"])
                    ),
                    "side": (
                        "LONG" if int(row["side_sign"]) == 1 else "SHORT"
                    ),
                    "relation": str(row["relation"]),
                    "inventory_mismatch_direction": _direction_token(
                        directions[0]
                    ),
                    "award_distortion_direction": _direction_token(
                        directions[1]
                    ),
                    "unmet_demand_mass_direction": _direction_token(
                        directions[2]
                    ),
                    "fee_distortion_direction": _direction_token(
                        directions[3]
                    ),
                    "agreement_count": int(row["agreement_count"]),
                    "prior_relation": str(row["prior_relation"]),
                }
            )
    frame = pd.DataFrame(rows, columns=CLOCK_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["entry_time", "control", "signal_id"], kind="mergesort"
        )
    if any(
        token in column.lower()
        for column in CLOCK_COLUMNS
        for token in FORBIDDEN_CLOCK_TOKENS
    ):
        raise RuntimeError("SCAF-48 clock schema exposes forbidden value")
    text = frame.to_csv(
        index=False,
        columns=CLOCK_COLUMNS,
        lineterminator="\n",
    ).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer,
        mode="wb",
        filename="",
        mtime=0,
    ) as zipped:
        zipped.write(text)
    return buffer.getvalue()


def _control_report(
    clocks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    primary = clocks["primary"]
    return {
        control: {
            "accepted_rows": len(clocks[control]),
            "train": clock_stats(_rows(clocks, control, "train")),
            "selection": clock_stats(_rows(clocks, control, "selection")),
            "same_side_reproduction_to_primary": {
                split: _same_side_reproduction(
                    _rows(clocks, "primary", split),
                    _rows(clocks, control, split),
                )
                for split in ("train", "selection")
            },
        }
        for control in prereg.CONTROL_ORDER
    }


def _core_payload(
    *,
    operations: pd.DataFrame,
    details: pd.DataFrame,
    features: Sequence[BatchFeature],
    raw: Mapping[str, Sequence[Mapping[str, Any]]],
    transitions: Sequence[Transition],
    clocks: Mapping[str, Sequence[Mapping[str, Any]]],
    preregistration: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    clock_bytes: bytes,
    clock_path: str | Path,
    artifact_eligible: bool,
    protocol_git_subprocess_calls: int,
) -> dict[str, Any]:
    (
        support_report,
        source_checks,
        composition,
        composition_checks,
    ) = support_and_composition(
        features,
        transitions,
        raw,
        clocks,
    )
    source_passed = all(source_checks.values())
    composition_passed = bool(
        source_passed and all(composition_checks.values())
    )
    novelty_report: dict[str, Any] = {}
    novelty_checks: dict[str, bool] = {}
    comparator_rows_decoded = 0
    comparator_status = "not_opened_source_support_or_composition_failed"
    if composition_passed and artifact_eligible:
        try:
            (
                novelty_report,
                novelty_checks,
                comparator_rows_decoded,
            ) = evaluate_novelty(clocks["primary"], preregistration)
            comparator_status = (
                "opened_after_complete_source_and_composition_pass"
            )
        except ComparatorContractFailure as error:
            comparator_rows_decoded = error.rows_decoded
            novelty_report = {
                "contract_failure": {
                    "code": error.code,
                    "message": str(error),
                    "comparator_rows_decoded": error.rows_decoded,
                }
            }
            novelty_checks = {
                f"comparator_contract:{error.code}": False
            }
            comparator_status = (
                "opened_after_complete_source_and_composition_pass_"
                "then_failed_closed"
            )
    elif composition_passed:
        comparator_status = "synthetic_build_not_authorized"
    novelty_passed = bool(
        composition_passed
        and artifact_eligible
        and novelty_checks
        and all(novelty_checks.values())
    )
    first_stage, first_check = first_failure(
        source_checks,
        composition_checks,
        novelty_checks,
        artifact_eligible=artifact_eligible,
    )
    if not source_passed or not composition_passed:
        decision = (
            "retire_SCAF_48_unchanged_before_comparators_and_outcomes"
        )
    elif not artifact_eligible:
        decision = "synthetic_build_cannot_authorize_comparators_or_outcomes"
    elif not novelty_passed:
        decision = "retire_SCAF_48_unchanged_before_outcomes"
    else:
        decision = "advance_to_separately_frozen_strict_economic_RLLM_evaluator"
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.Policy().policy_id,
        "artifact_eligible": artifact_eligible,
        "outcomes_opened": False,
        "post_entry_return_computed": False,
        "funding_loaded": False,
        "source_incidence_opened": True,
        "source_rows_decoded": len(operations) + len(details),
        "comparator_rows_decoded": comparator_rows_decoded,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
            "builder": {
                "path": str(PREREGISTRATION_BUILDER),
                "sha256": PREREGISTRATION_BUILDER_SHA256,
            },
        },
        "implementation": {
            "source": str(SCRIPT_PATH),
            "source_sha256": sha256_file(SCRIPT_PATH),
            "test": str(TEST_PATH),
            "test_sha256": sha256_file(TEST_PATH),
            "contract": str(IMPLEMENTATION_CONTRACT),
            "contract_sha256": IMPLEMENTATION_CONTRACT_SHA256,
            "committed_clean_before_real_source": artifact_eligible,
        },
        "source_audit": dict(source_audit),
        "feature_funnel": {
            "operation_rows": len(operations),
            "detail_rows": len(details),
            "causal_batches": len(features),
            "valid_batches": sum(feature.valid for feature in features),
            "invalid_batches": sum(not feature.valid for feature in features),
            "valid_transitions": len(transitions),
            "raw_primary_opportunities": len(raw["primary"]),
            "accepted_primary_rows": len(clocks["primary"]),
        },
        "invalid_batch_reasons": dict(
            Counter(
                feature.invalid_reason
                for feature in features
                if feature.invalid_reason is not None
            )
        ),
        "controls": _control_report(clocks),
        "source_support": support_report,
        "source_support_checks": source_checks,
        "source_support_passed": source_passed,
        "relational_composition": composition,
        "relational_composition_checks": composition_checks,
        "relational_composition_passed": composition_passed,
        "comparator_status": comparator_status,
        "novelty": novelty_report,
        "novelty_checks": novelty_checks,
        "novelty_passed": novelty_passed,
        "first_failing_stage": first_stage,
        "first_failing_check": first_check,
        "decision": decision,
        "clock": {
            "path": str(clock_path),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": sum(len(clocks[name]) for name in prereg.CONTROL_ORDER),
            "columns": list(CLOCK_COLUMNS),
            "control_counts": {
                name: len(clocks[name]) for name in prereg.CONTROL_ORDER
            },
            "deterministic_gzip_mtime_zero": True,
        },
        "outcome_boundary": {
            "btc_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "future_return_rows_computed": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_source_rows_loaded": 0,
            "network_calls": 0,
            "external_data_subprocess_calls": 0,
            "protocol_git_subprocess_calls": protocol_git_subprocess_calls,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def _build_support(
    operations: pd.DataFrame,
    details: pd.DataFrame,
    *,
    preregistration: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    clock_path: str | Path,
    artifact_eligible: bool,
    protocol_git_subprocess_calls: int,
) -> tuple[dict[str, Any], bytes]:
    features = build_batch_features(operations, details)
    raw, transitions = build_raw_candidates(features)
    clocks = build_clocks(raw)
    clock_bytes = deterministic_clock_bytes(clocks)
    report = _core_payload(
        operations=operations,
        details=details,
        features=features,
        raw=raw,
        transitions=transitions,
        clocks=clocks,
        preregistration=preregistration,
        source_audit=source_audit,
        clock_bytes=clock_bytes,
        clock_path=clock_path,
        artifact_eligible=artifact_eligible,
        protocol_git_subprocess_calls=protocol_git_subprocess_calls,
    )
    return report, clock_bytes


def build_support_from_frames(
    operations: pd.DataFrame,
    details: pd.DataFrame,
) -> tuple[dict[str, Any], bytes]:
    validated_operations = validate_operations(operations)
    validated_details = validate_details(details)
    reconcile_source(validated_operations, validated_details)
    return _build_support(
        validated_operations,
        validated_details,
        preregistration=prereg.build_manifest(),
        source_audit={
            "kind": "synthetic_or_injected",
            "operation_rows": len(validated_operations),
            "detail_rows": len(validated_details),
            "bindings": {},
        },
        clock_path=DEFAULT_CLOCK_OUTPUT,
        artifact_eligible=False,
        protocol_git_subprocess_calls=0,
    )


def validate_report(payload: Mapping[str, Any]) -> None:
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("SCAF-48 report manifest hash mismatch")
    for name in (
        "outcomes_opened",
        "post_entry_return_computed",
        "funding_loaded",
    ):
        if payload.get(name) is not False:
            raise RuntimeError(f"SCAF-48 report opened forbidden evidence: {name}")
    boundary = payload["outcome_boundary"]
    for name in (
        "btc_market_rows_loaded",
        "funding_rows_loaded",
        "future_return_rows_computed",
        "return_or_pnl_fields_read",
        "post_2023_source_rows_loaded",
        "network_calls",
        "external_data_subprocess_calls",
    ):
        if boundary.get(name) != 0:
            raise RuntimeError(f"SCAF-48 forbidden evidence opened: {name}")
    if (
        not payload["source_support_passed"]
        or not payload["relational_composition_passed"]
    ) and payload["comparator_rows_decoded"] != 0:
        raise RuntimeError("SCAF-48 comparator opened before source pass")


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


def _output_relative(path: str | Path) -> Path:
    candidate = Path(path)
    raw = str(path)
    if (
        candidate.is_absolute()
        or raw.startswith("~")
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("SCAF-48 output must be repository-relative")
    return candidate


def _open_parent(candidate: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    current = os.open(REPOSITORY_ROOT, flags)
    try:
        for part in candidate.parent.parts:
            next_descriptor = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_descriptor
        return current
    except Exception:
        os.close(current)
        raise


def _read_regular(directory: int, name: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RuntimeError("SCAF-48 output path is unsafe") from error
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("SCAF-48 output is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_once(path: str | Path, payload: bytes) -> str:
    output = _output_relative(path)
    parent = _open_parent(output)
    temporary = f".{output.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    temporary_created = False
    try:
        try:
            existing = _read_regular(parent, output.name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != payload:
                raise RuntimeError("SCAF-48 existing artifact is noncanonical")
            return "verified_existing"
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        temporary_created = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary,
                output.name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_regular(parent, output.name) != payload:
                raise RuntimeError("SCAF-48 artifact race drift")
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
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> dict[str, Any]:
    _assert_protocol_committed()
    preregistration = validate_preregistration()
    bindings = verify_pre_source_bindings(preregistration)
    operations, details = load_sources()
    report, clock_bytes = _build_support(
        operations,
        details,
        preregistration=preregistration,
        source_audit={
            "kind": "frozen_repository_sources",
            "operation_rows": len(operations),
            "detail_rows": len(details),
            "bindings": bindings,
            "operation_allowlist": list(prereg.OPERATIONS_ALLOWLIST),
            "detail_allowlist": list(prereg.DETAILS_ALLOWLIST),
        },
        clock_path=clock_output,
        artifact_eligible=True,
        protocol_git_subprocess_calls=2,
    )
    report_bytes = canonical_report_bytes(report)
    _write_once(clock_output, clock_bytes)
    _write_once(report_output, report_bytes)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT))
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK_OUTPUT))
    arguments = parser.parse_args()
    report = run(
        report_output=arguments.report_output,
        clock_output=arguments.clock_output,
    )
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
