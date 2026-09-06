"""Evaluate outcome-blind source support for the frozen TUSI-168 policy.

This module consumes only the promoted TRON USDT supply-event CSV and its
manifest.  It does not import, discover, or open comparator, market, funding,
return, or outcome artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import secrets
import stat
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

import pandas as pd

from training import build_tron_usdt_supply_events as source_builder
from training import preregister_tron_usdt_supply_impulse as prereg


def _timestamp(value: Any) -> pd.Timestamp:
    """Narrow pandas' Timestamp | NaTType constructor result."""

    timestamp = pd.Timestamp(value)
    if not isinstance(timestamp, pd.Timestamp):
        raise RuntimeError("TUSI-168 timestamp is NaT")
    return timestamp


def _epoch_timestamp(seconds: int) -> pd.Timestamp:
    timestamp = pd.Timestamp(seconds, unit="s", tz="UTC")
    if not isinstance(timestamp, pd.Timestamp):
        raise RuntimeError("TUSI-168 epoch timestamp is NaT")
    return timestamp


def _timedelta(seconds: int) -> pd.Timedelta:
    duration = pd.Timedelta(seconds=seconds)
    if not isinstance(duration, pd.Timedelta):
        raise RuntimeError("TUSI-168 duration is NaT")
    return duration


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Narrow pandas' duplicate-column DataFrame | Series indexing union."""

    values = frame[column]
    if not isinstance(values, pd.Series):
        raise RuntimeError(f"TUSI-168 duplicate or ambiguous column: {column}")
    return values


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Typed adapter for pandas' dynamically typed records orientation."""

    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"TUSI-168 noninteger value: {label}")
    return value


POLICY_ID = "TUSI-168"
PROTOCOL_VERSION = "tron_usdt_supply_impulse_source_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/evaluate_tron_usdt_supply_impulse_source_support.py")
TEST_PATH = Path("tests/test_evaluate_tron_usdt_supply_impulse_source_support.py")

DEFAULT_SOURCE_CSV = source_builder.DEFAULT_CSV_OUTPUT
DEFAULT_SOURCE_MANIFEST = source_builder.DEFAULT_MANIFEST_OUTPUT
DEFAULT_PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "54817044b8df76dc347ed64b6fe5f6f2dfdddcdb211bded4ba2b1af133d49067"
)
PREREGISTRATION_MANIFEST_HASH = (
    "d67cd1b67632ae92e9458395e729627a6f4c3b4b75ce97187653eac3a09e40c1"
)
DEFAULT_PRIMARY_OUTPUT = Path(
    "results/tron_usdt_supply_impulse_primary_clock_2026-07-30.csv.gz"
)
DEFAULT_CONTROLS_OUTPUT = Path(
    "results/tron_usdt_supply_impulse_control_clocks_2026-07-30.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/tron_usdt_supply_impulse_source_support_2026-07-30.json"
)

SOURCE_COLUMNS = tuple(source_builder.CSV_COLUMNS)
CLOCK_COLUMNS = (
    "policy_id",
    "control",
    "window",
    "constituent_identities_json",
    "source_identity",
    "constituent_count",
    "bucket_amount_raw",
    "decision_time_utc",
    "entry_time_utc",
    "exit_time_utc",
    "side",
)
INDEPENDENT_CONTROLS = (
    "primary",
    "issue_only",
    "redeem_only",
    "include_destroyed_black_funds",
    "count_net_side",
)
SAME_PARENT_CONTROLS = (
    "exact_direction_flip",
    "deterministic_random_side",
    "constant_long",
    "constant_short",
    "one_bar_delayed_entry",
)
CONTROL_ORDER = INDEPENDENT_CONTROLS + SAME_PARENT_CONTROLS

BAR_SECONDS = 300
HOLD_SECONDS = 168 * 3600
BAR = _timedelta(BAR_SECONDS)
HOLD = _timedelta(HOLD_SECONDS)
UTC = "UTC"
FULL_START = _timestamp("2023-06-01T00:00:00Z")
FULL_END = _timestamp("2026-06-01T00:00:00Z")
SPLITS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "selection": (
        _timestamp("2023-06-01T00:00:00Z"),
        _timestamp("2025-01-01T00:00:00Z"),
    ),
    "future25": (
        _timestamp("2025-01-01T00:00:00Z"),
        _timestamp("2026-01-01T00:00:00Z"),
    ),
    "future26": (
        _timestamp("2026-01-01T00:00:00Z"),
        _timestamp("2026-06-01T00:00:00Z"),
    ),
}
HALF_YEARS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "2023H2": (
        _timestamp("2023-06-01T00:00:00Z"),
        _timestamp("2024-01-01T00:00:00Z"),
    ),
    "2024H1": (
        _timestamp("2024-01-01T00:00:00Z"),
        _timestamp("2024-07-01T00:00:00Z"),
    ),
    "2024H2": (
        _timestamp("2024-07-01T00:00:00Z"),
        _timestamp("2025-01-01T00:00:00Z"),
    ),
    "2025H1": (
        _timestamp("2025-01-01T00:00:00Z"),
        _timestamp("2025-07-01T00:00:00Z"),
    ),
    "2025H2": (
        _timestamp("2025-07-01T00:00:00Z"),
        _timestamp("2026-01-01T00:00:00Z"),
    ),
}
DIAGNOSTIC_PERIODS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "full": (FULL_START, FULL_END),
    "2024": (
        _timestamp("2024-01-01T00:00:00Z"),
        _timestamp("2025-01-01T00:00:00Z"),
    ),
    **HALF_YEARS,
}
SUPPORT_FLOORS = {
    "selection": 8,
    "2023H2": 2,
    "2024H1": 2,
    "2024H2": 2,
    "future25": 4,
    "2025H1": 1,
    "2025H2": 1,
    "future26": 2,
}

EVENT_DIRECTIONS = {
    "Issue": 1,
    "Redeem": -1,
    "DestroyedBlackFunds": -1,
}
HEX_64 = re.compile(r"0x[0-9a-f]{64}\Z")
HEX_40 = re.compile(r"0x[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
FORBIDDEN_TOKENS = (
    "price",
    "return",
    "label",
    "target",
    "outcome",
    "funding",
    "premium",
    "market",
    "pnl",
    "cagr",
    "mdd",
    "reward",
    "comparator",
    "gross9",
    "btc",
)
EVIDENCE_BOUNDARY = {
    "source_rows_opened": 0,
    "comparator_rows_opened": 0,
    "gross9_rows_opened": 0,
    "btc_market_rows_opened": 0,
    "funding_rows_opened": 0,
    "outcome_rows_opened": 0,
    "outcomes_computed": False,
    "network_calls": 0,
}
PERIOD_ORDER = (
    "selection",
    "2023H2",
    "2024",
    "2024H1",
    "2024H2",
    "future25",
    "2025H1",
    "2025H2",
    "future26",
    "full",
)
SOURCE_CONTRACT_AUDIT_KEYS = (
    "artifact_eligible",
    "source_csv_path",
    "source_csv_sha256",
    "source_csv_bytes",
    "source_manifest_path",
    "source_manifest_sha256",
    "source_manifest_hash",
    "dual_raw_log_replay_differences",
    "chunk_integrity_differences",
    "receipt_header_differences",
    "issue_mint_transfer_pair_differences",
    "redeem_burn_transfer_pair_differences",
    "deprecate_events",
    "source_integrity",
)
REPORT_KEYS = {
    "protocol_version",
    "policy_id",
    "status",
    "terminal",
    "artifact_eligible",
    "support_passed",
    "decision",
    "registration",
    "source_contract",
    "raw_candidate_counts",
    "accepted_clock_counts",
    "period_diagnostics",
    "support_audit",
    "support_checks",
    "future_append_selection_invariance",
    "control_overlap",
    "clock_artifacts",
    "evidence_boundary",
    "source_support_precedes_novelty",
    "novelty_comparator_market_or_outcome_artifacts_opened",
    "manifest_hash",
}


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_time(value: Any) -> str:
    timestamp = _timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("TUSI-168 timestamps must be timezone-aware")
    timestamp = timestamp.tz_convert(UTC)
    if timestamp.microsecond or timestamp.nanosecond:
        raise RuntimeError("TUSI-168 timestamps must be whole-second")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(series: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="raise")
    if parsed.isna().any():
        raise RuntimeError(f"TUSI-168 null timestamp: {name}")
    if any(value.microsecond or value.nanosecond for value in parsed):
        raise RuntimeError(f"TUSI-168 subsecond timestamp: {name}")
    return parsed


def _reject_forbidden_columns(columns: Iterable[Any]) -> None:
    for column in columns:
        lowered = str(column).lower()
        if any(token in lowered for token in FORBIDDEN_TOKENS):
            raise RuntimeError(f"TUSI-168 outcome-like column rejected: {column}")


def _exact_integer(
    series: pd.Series,
    name: str,
    *,
    positive: bool = False,
    int64: bool = True,
) -> pd.Series:
    def convert(value: Any) -> int:
        if isinstance(value, bool) or value is None or pd.isna(value):
            raise RuntimeError(f"TUSI-168 noninteger source field: {name}")
        text = str(value)
        if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text):
            raise RuntimeError(f"TUSI-168 noninteger source field: {name}")
        result = int(text)
        if positive and result <= 0:
            raise RuntimeError(f"TUSI-168 nonpositive source field: {name}")
        return result

    converted = series.map(convert)
    if int64:
        if any(value < -(2**63) or value >= 2**63 for value in converted):
            raise RuntimeError(f"TUSI-168 int64 overflow: {name}")
        return converted.astype("int64")
    return converted.astype("object")


def validate_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the normalized source-only CSV contract and row invariants."""

    if list(frame.columns) != list(SOURCE_COLUMNS):
        _reject_forbidden_columns(frame.columns)
        raise RuntimeError("TUSI-168 source exact schema drift")
    _reject_forbidden_columns(frame.columns)
    if frame.empty:
        raise RuntimeError("TUSI-168 source is empty")
    rows = frame.copy()
    canonical_timestamp = re.compile(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
        + r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z"
    )
    for column in ("event_timestamp_utc", "available_at_utc"):
        values = _series(rows, column)
        if not bool(values.map(
            lambda value: isinstance(value, str)
            and canonical_timestamp.fullmatch(value) is not None
        ).all()):
            raise RuntimeError(
                f"TUSI-168 noncanonical whole-second UTC string: {column}"
            )
        for value in values:
            if _format_time(value) != value:
                raise RuntimeError(
                    f"TUSI-168 noncanonical UTC timestamp: {column}"
                )
    for column in (
        "amount_raw",
        "block_number",
        "transaction_index",
        "log_index",
        "confirmation_block",
    ):
        rows[column] = _exact_integer(
            _series(rows, column),
            column,
            positive=column == "amount_raw",
            int64=column != "amount_raw",
        )
    rows["supply_direction"] = _exact_integer(
        _series(rows, "supply_direction"), "supply_direction"
    )
    if bool(
        _series(rows, "amount_raw")
        .map(lambda value: int(value) >= 2**256)
        .any()
    ):
        raise RuntimeError("TUSI-168 amount_raw must be below 2**256")
    rows["paired_transfer_log_index"] = _series(
        rows, "paired_transfer_log_index"
    ).map(
        lambda value: (
            None
            if value is None or pd.isna(value) or str(value) == ""
            else (
                int(value)
                if isinstance(value, float) and value.is_integer()
                else int(
                    _exact_integer(
                        pd.Series([value]), "paired_transfer_log_index"
                    ).iloc[0]
                )
            )
        )
    )
    for column in ("event_timestamp_utc", "available_at_utc"):
        rows[column] = _parse_time(_series(rows, column), column)

    if not bool(_series(rows, "event_type").isin(EVENT_DIRECTIONS).all()):
        raise RuntimeError("TUSI-168 unsupported event type or Deprecate")
    if _series(rows, "supply_direction").tolist() != [
        EVENT_DIRECTIONS[str(value)] for value in _series(rows, "event_type")
    ]:
        raise RuntimeError("TUSI-168 supply direction drift")
    if (
        not bool(_series(rows, "block_hash")
        .map(lambda value: bool(HEX_64.fullmatch(str(value))))
        .all())
    ):
        raise RuntimeError("TUSI-168 malformed block hash")
    if (
        not bool(_series(rows, "transaction_hash")
        .map(lambda value: bool(HEX_64.fullmatch(str(value))))
        .all())
    ):
        raise RuntimeError("TUSI-168 malformed transaction hash")
    if (
        not bool(_series(rows, "confirmation_block_hash")
        .map(lambda value: bool(HEX_64.fullmatch(str(value))))
        .all())
    ):
        raise RuntimeError("TUSI-168 malformed confirmation block hash")
    if (
        not bool(_series(rows, "actor_address")
        .map(lambda value: bool(HEX_40.fullmatch(str(value))))
        .all())
    ):
        raise RuntimeError("TUSI-168 malformed actor address")
    if not (
        _series(rows, "confirmation_block")
        == _series(rows, "block_number") + source_builder.CONFIRMATION_BLOCKS
    ).all():
        raise RuntimeError("TUSI-168 confirmation block drift")
    if bool(
        (
            _series(rows, "available_at_utc")
            <= _series(rows, "event_timestamp_utc")
        ).any()
    ):
        raise RuntimeError("TUSI-168 availability does not follow event time")
    if (
        bool(
            _series(rows, "block_number")
            .lt(source_builder.SOURCE_START_BLOCK)
            .any()
        )
        or bool(
            _series(rows, "block_number")
            .gt(source_builder.LAST_EVENT_BLOCK)
            .any()
        )
    ):
        raise RuntimeError("TUSI-168 event block outside frozen source range")
    for column in (
        "transaction_index",
        "log_index",
        "paired_transfer_log_index",
    ):
        if bool(_series(rows, column).dropna().map(int).lt(0).any()):
            raise RuntimeError(f"TUSI-168 negative source field: {column}")
    source_start = _timestamp(source_builder.SOURCE_START_UTC)
    source_end = _timestamp(source_builder.END_BOUNDARY_UTC)
    if (
        bool(_series(rows, "event_timestamp_utc").lt(source_start).any())
        or bool(_series(rows, "event_timestamp_utc").ge(source_end).any())
        or bool(_series(rows, "available_at_utc").ge(source_end).any())
    ):
        raise RuntimeError("TUSI-168 source timestamp outside frozen range")

    paired_expected = _series(rows, "event_type").isin(("Issue", "Redeem"))
    paired_actual = _series(rows, "paired_transfer_log_index").notna()
    if not bool(paired_actual.eq(paired_expected).all()):
        raise RuntimeError("TUSI-168 semantic/Transfer pairing shape drift")
    if (
        rows.loc[paired_actual, "paired_transfer_log_index"].map(int)
        == rows.loc[paired_actual, "log_index"]
    ).any():
        raise RuntimeError("TUSI-168 semantic event paired to itself")

    identity_columns = [
        "block_number",
        "transaction_index",
        "log_index",
        "transaction_hash",
    ]
    if rows.duplicated(identity_columns).any():
        raise RuntimeError("TUSI-168 duplicate canonical event identity")
    expected_order = rows.sort_values(identity_columns, kind="mergesort").index
    if not expected_order.equals(rows.index):
        raise RuntimeError("TUSI-168 source row order drift")
    if (
        not rows["event_timestamp_utc"].is_monotonic_increasing
        or not rows["available_at_utc"].is_monotonic_increasing
    ):
        raise RuntimeError("TUSI-168 source timestamps decrease in block order")
    return rows.reset_index(drop=True)


def candidate_entry_time(available_at: Any) -> pd.Timestamp:
    timestamp = _timestamp(available_at)
    if timestamp.tzinfo is None:
        raise RuntimeError("TUSI-168 availability must be timezone-aware")
    timestamp = timestamp.tz_convert(UTC)
    if timestamp.microsecond or timestamp.nanosecond:
        raise RuntimeError("TUSI-168 availability must be whole-second")
    seconds = int(timestamp.timestamp())
    entry_seconds = ((seconds + BAR_SECONDS - 1) // BAR_SECONDS) * BAR_SECONDS
    return _epoch_timestamp(entry_seconds + BAR_SECONDS)


def _constituent_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["block_number"]),
        int(row["transaction_index"]),
        int(row["log_index"]),
        str(row["transaction_hash"]),
        str(row["event_type"]),
        int(row["amount_raw"]),
    )


def canonical_constituent_bytes(rows: pd.DataFrame) -> bytes:
    identities = sorted(_constituent_identity(row) for row in _frame_records(rows))
    return json.dumps(
        identities,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def source_identity(rows: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_constituent_bytes(rows)).hexdigest()


def _empty_clock() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(CLOCK_COLUMNS))


def _raw_candidates_from_validated(
    frame: pd.DataFrame, control: str
) -> pd.DataFrame:
    if control not in INDEPENDENT_CONTROLS:
        raise ValueError("TUSI-168 raw candidates require an independent control")
    rows = frame.copy()
    if control == "issue_only":
        rows = rows.loc[rows["event_type"].eq("Issue")].copy()
    elif control == "redeem_only":
        rows = rows.loc[rows["event_type"].eq("Redeem")].copy()
    elif control != "include_destroyed_black_funds":
        rows = rows.loc[rows["event_type"].isin(("Issue", "Redeem"))].copy()
    if rows.empty:
        return _empty_clock()
    rows["_entry"] = rows["available_at_utc"].map(candidate_entry_time)

    output: list[dict[str, Any]] = []
    for entry, bucket in rows.groupby("_entry", sort=True):
        entry_time = _timestamp(entry)
        issue = bucket.loc[bucket["event_type"].eq("Issue")]
        redeem = bucket.loc[bucket["event_type"].eq("Redeem")]
        destroyed = bucket.loc[bucket["event_type"].eq("DestroyedBlackFunds")]
        amount = int(issue["amount_raw"].sum()) - int(redeem["amount_raw"].sum())
        if control == "include_destroyed_black_funds":
            amount -= int(destroyed["amount_raw"].sum())
        if control == "issue_only":
            amount = int(issue["amount_raw"].sum())
        elif control == "redeem_only":
            amount = -int(redeem["amount_raw"].sum())
        signal = len(issue) - len(redeem) if control == "count_net_side" else amount
        if signal == 0:
            continue
        side = "LONG" if signal > 0 else "SHORT"
        decision = _timestamp(_series(bucket, "available_at_utc").max())
        output.append(
            {
                "policy_id": POLICY_ID,
                "control": control,
                "window": None,
                "constituent_identities_json": canonical_constituent_bytes(
                    bucket
                ).decode("utf-8"),
                "source_identity": source_identity(bucket),
                "constituent_count": len(bucket),
                "bucket_amount_raw": int(signal),
                "decision_time_utc": decision,
                "entry_time_utc": entry_time,
                "exit_time_utc": entry_time + HOLD,
                "side": side,
            }
        )
    candidates = pd.DataFrame(output, columns=pd.Index(CLOCK_COLUMNS))
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        ["entry_time_utc", "decision_time_utc", "source_identity", "side"],
        kind="mergesort",
    ).reset_index(drop=True)


def raw_candidates(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    """Build exact-entry buckets independently for one frozen control."""

    return _raw_candidates_from_validated(validate_source_frame(frame), control)


def _assign_window(rows: pd.DataFrame) -> pd.DataFrame:
    assigned: list[pd.Series] = []
    for _, row in rows.iterrows():
        if not (
            row["entry_time_utc"] >= FULL_START and row["exit_time_utc"] <= FULL_END
        ):
            continue
        for name, (start, end) in SPLITS.items():
            if row["entry_time_utc"] >= start and row["exit_time_utc"] <= end:
                copied = row.copy()
                copied["window"] = name
                assigned.append(copied)
                break
    return (
        pd.DataFrame(assigned, columns=pd.Index(CLOCK_COLUMNS))
        if assigned
        else _empty_clock()
    )


def reserve_nonoverlap(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply containment and one global non-overlap schedule."""

    contained = _assign_window(rows)
    if contained.empty:
        return contained
    ordered = contained.sort_values(
        ["entry_time_utc", "decision_time_utc", "source_identity", "side"],
        kind="mergesort",
    )
    accepted: list[pd.Series] = []
    prior_exit: pd.Timestamp | None = None
    for _, row in ordered.iterrows():
        if prior_exit is None or row["entry_time_utc"] >= prior_exit:
            accepted.append(row)
            prior_exit = _timestamp(row["exit_time_utc"])
    return pd.DataFrame(
        accepted, columns=pd.Index(CLOCK_COLUMNS)
    ).reset_index(drop=True)


def deterministic_random_side(identity: str) -> str:
    digest = hashlib.sha256(f"{identity}|{POLICY_ID}|RANDOM_SIDE".encode()).digest()
    return "LONG" if digest[0] < 128 else "SHORT"


def _parent_control(primary: pd.DataFrame, control: str) -> pd.DataFrame:
    if control not in SAME_PARENT_CONTROLS:
        raise ValueError(f"TUSI-168 unknown parent control: {control}")
    rows = primary.copy()
    rows["control"] = control
    if control == "exact_direction_flip":
        rows["side"] = _series(rows, "side").map(
            lambda side: {"LONG": "SHORT", "SHORT": "LONG"}[str(side)]
        )
    elif control == "deterministic_random_side":
        rows["side"] = rows["source_identity"].map(deterministic_random_side)
    elif control == "constant_long":
        rows["side"] = "LONG"
    elif control == "constant_short":
        rows["side"] = "SHORT"
    else:
        rows["entry_time_utc"] = rows["entry_time_utc"] + BAR
        rows["exit_time_utc"] = rows["exit_time_utc"] + BAR
    return rows.loc[:, CLOCK_COLUMNS].reset_index(drop=True)


def _build_controls_from_validated(
    frame: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    controls: dict[str, pd.DataFrame] = {}
    raw_counts: dict[str, int] = {}
    for control in INDEPENDENT_CONTROLS:
        raw = _raw_candidates_from_validated(frame, control)
        raw_counts[control] = len(raw)
        controls[control] = reserve_nonoverlap(raw)
    for control in SAME_PARENT_CONTROLS:
        raw_counts[control] = len(controls["primary"])
        controls[control] = _parent_control(controls["primary"], control)
    return controls, raw_counts


def build_controls(
    frame: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    return _build_controls_from_validated(validate_source_frame(frame))


def _csv_bytes(frame: pd.DataFrame, *, compress: bool = True) -> bytes:
    serial = frame.loc[:, CLOCK_COLUMNS].copy()
    for column in ("decision_time_utc", "entry_time_utc", "exit_time_utc"):
        serial[column] = serial[column].map(_format_time)
    raw = serial.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if not compress:
        return raw
    return _deterministic_gzip(raw)


def _deterministic_gzip(raw: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer,
        mode="wb",
        filename="",
        compresslevel=9,
        mtime=0,
    ) as handle:
        handle.write(raw)
    return buffer.getvalue()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
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


FUTURE_APPEND_ROW_KEYS = {
    "accepted",
    "control",
    "constituent_identities",
    "source_identity",
    "constituent_count",
    "signed_bucket_amount_or_count",
    "decision_time_utc",
    "entry_time_utc",
    "exit_time_utc",
    "side",
}


def _append_view_payload(
    rows: pd.DataFrame, *, accepted: bool
) -> list[dict[str, Any]]:
    return [
        {
            "accepted": accepted,
            "control": str(row["control"]),
            "constituent_identities": json.loads(
                str(row["constituent_identities_json"])
            ),
            "source_identity": str(row["source_identity"]),
            "constituent_count": int(row["constituent_count"]),
            "signed_bucket_amount_or_count": int(row["bucket_amount_raw"]),
            "decision_time_utc": _format_time(row["decision_time_utc"]),
            "entry_time_utc": _format_time(row["entry_time_utc"]),
            "exit_time_utc": _format_time(row["exit_time_utc"]),
            "side": str(row["side"]),
        }
        for row in _frame_records(rows)
    ]


def _selection_construction_views(
    frame: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build every frozen construction once and project its selection views."""

    views: dict[str, dict[str, pd.DataFrame]] = {}
    accepted: dict[str, pd.DataFrame] = {}
    for control in INDEPENDENT_CONTROLS:
        raw = _raw_candidates_from_validated(frame, control)
        assigned = _assign_window(raw)
        accepted[control] = reserve_nonoverlap(raw)
        views[control] = {
            "raw": assigned.loc[assigned["window"].eq("selection")].reset_index(
                drop=True
            ),
            "accepted": accepted[control]
            .loc[accepted[control]["window"].eq("selection")]
            .reset_index(drop=True),
        }
    for control in SAME_PARENT_CONTROLS:
        parent = _parent_control(accepted["primary"], control)
        views[control] = {
            "accepted": parent.loc[parent["window"].eq("selection")].reset_index(
                drop=True
            )
        }
    return views


def future_append_selection_invariance(
    frame: pd.DataFrame,
) -> tuple[bool, dict[str, Any]]:
    """Compare two fresh full/prefix builds for every frozen construction."""

    validated = validate_source_frame(frame)
    selection_end = SPLITS["selection"][1]
    prefix = validated.loc[validated["available_at_utc"].lt(selection_end)].reset_index(
        drop=True
    )
    full_views = _selection_construction_views(validated)
    if prefix.empty:
        prefix_views = {
            control: {
                view_name: _empty_clock()
                for view_name in (
                    ("raw", "accepted")
                    if control in INDEPENDENT_CONTROLS
                    else ("accepted",)
                )
            }
            for control in CONTROL_ORDER
        }
    else:
        prefix_views = _selection_construction_views(prefix)
    comparisons: dict[str, Any] = {}
    total_differences = 0
    for control in CONTROL_ORDER:
        comparisons[control] = {}
        view_names = (
            ("raw", "accepted") if control in INDEPENDENT_CONTROLS else ("accepted",)
        )
        for view_name in view_names:
            accepted_view = view_name == "accepted"
            full_payload = _append_view_payload(
                full_views[control][view_name],
                accepted=accepted_view,
            )
            prefix_payload = _append_view_payload(
                prefix_views[control][view_name],
                accepted=accepted_view,
            )
            equal = full_payload == prefix_payload
            differences = 0 if equal else 1
            total_differences += differences
            comparisons[control][view_name] = {
                "differences": differences,
                "full_rows": len(full_payload),
                "prefix_rows": len(prefix_payload),
                "full_sha256": canonical_hash(full_payload),
                "prefix_sha256": canonical_hash(prefix_payload),
            }
    return total_differences == 0, {
        "prefix_source_rule": ("available_at_utc < 2025-01-01T00:00:00Z"),
        "total_differences": total_differences,
        "comparisons": comparisons,
    }


def project_clock_to_period(
    primary: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Project an accepted clock only after rechecking shifted containment."""

    return primary.loc[
        primary["entry_time_utc"].ge(start) & primary["exit_time_utc"].le(end)
    ]


def _period_diagnostic(rows: pd.DataFrame) -> dict[str, Any]:
    months = (
        rows["entry_time_utc"].dt.strftime("%Y-%m").value_counts().sort_index()
        if not rows.empty
        else pd.Series(dtype="int64")
    )
    total = len(rows)
    maximum = int(months.max()) if total else 0
    return {
        "accepted": total,
        "LONG": int(rows["side"].eq("LONG").sum()),
        "SHORT": int(rows["side"].eq("SHORT").sum()),
        "utc_entry_month_counts": {
            str(month): int(count) for month, count in months.items()
        },
        "maximum_month_count": maximum,
        "maximum_month_share_exact": (
            {"numerator": maximum, "denominator": total} if total else None
        ),
    }


def exact_entry_jaccard(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, int]:
    a = {int(value.timestamp()) for value in left["entry_time_utc"]}
    b = {int(value.timestamp()) for value in right["entry_time_utc"]}
    union = a | b
    numerator, denominator = (len(a & b), len(union)) if union else (0, 1)
    return {"numerator": numerator, "denominator": denominator}


def control_overlap(controls: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    primary = controls["primary"]
    primary_entries = set(primary["entry_time_utc"])
    report: dict[str, Any] = {}
    for name in CONTROL_ORDER[1:]:
        rows = controls[name]
        entries = set(rows["entry_time_utc"])
        report[name] = {
            "accepted": len(rows),
            "exact_entry_intersection": len(primary_entries & entries),
            "exact_entry_union": len(primary_entries | entries),
            "exact_entry_jaccard": exact_entry_jaccard(primary, rows),
            "exact_parent_identity_intersection": len(
                set(primary["source_identity"]) & set(rows["source_identity"])
            ),
        }
    return report


def _clock_reproducible(primary: pd.DataFrame) -> bool:
    if primary.empty or primary["source_identity"].duplicated().any():
        return False
    ordered = primary.sort_values(
        ["entry_time_utc", "decision_time_utc", "source_identity", "side"],
        kind="mergesort",
    ).reset_index(drop=True)
    if not ordered.equals(primary.reset_index(drop=True)):
        return False
    for row in _frame_records(primary):
        try:
            constituent_bytes = str(row["constituent_identities_json"]).encode(
                "utf-8"
            )
            constituents = json.loads(constituent_bytes)
        except (UnicodeEncodeError, json.JSONDecodeError):
            return False
        if (
            not SHA256.fullmatch(str(row["source_identity"]))
            or hashlib.sha256(constituent_bytes).hexdigest()
            != row["source_identity"]
            or not isinstance(constituents, list)
            or len(constituents) != row["constituent_count"]
            or constituents != sorted(constituents)
            or json.dumps(
                constituents,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            != constituent_bytes
            or not isinstance(row["bucket_amount_raw"], int)
            or row["bucket_amount_raw"] == 0
            or (
                "LONG" if row["bucket_amount_raw"] > 0 else "SHORT"
            )
            != row["side"]
            or row["side"] not in {"LONG", "SHORT"}
            or row["exit_time_utc"] != row["entry_time_utc"] + HOLD
            or row["entry_time_utc"].second != 0
            or row["entry_time_utc"].minute % 5
            or row["decision_time_utc"] >= row["entry_time_utc"]
        ):
            return False
    return True


def support_checks(
    primary: pd.DataFrame,
    *,
    source_audit: Mapping[str, Any] | None = None,
    append_invariance_passed: bool = True,
    reproducible: bool = True,
) -> tuple[dict[str, Any], dict[str, bool]]:
    audit_values = dict(source_audit or {})
    periods = {
        name: _period_diagnostic(project_clock_to_period(primary, start, end))
        for name, (start, end) in {
            **SPLITS,
            **DIAGNOSTIC_PERIODS,
        }.items()
    }
    ordered = primary.sort_values("entry_time_utc")
    gaps = ordered["entry_time_utc"].diff().dropna()
    max_gap = int(gaps.max().total_seconds()) if not gaps.empty else None

    source_gate_names = (
        "dual_raw_log_replay_differences",
        "chunk_integrity_differences",
        "receipt_header_differences",
        "issue_mint_transfer_pair_differences",
        "redeem_burn_transfer_pair_differences",
        "deprecate_events",
    )
    checks = {
        f"source_{name}_zero": int(audit_values.get(name, 0)) == 0
        for name in source_gate_names
    }
    manifest_integrity = audit_values.get(
        "source_integrity", source_builder.ZERO_SOURCE_INTEGRITY
    )
    if not isinstance(manifest_integrity, Mapping) or set(manifest_integrity) != set(
        source_builder.ZERO_SOURCE_INTEGRITY
    ):
        raise RuntimeError("TUSI-168 source integrity audit schema drift")
    checks.update(
        {
            f"manifest_source_integrity_{name}_zero": _integer(
                value, f"source_integrity.{name}"
            )
            == 0
            for name, value in manifest_integrity.items()
        }
    )
    checks["future_append_selection_differences_zero"] = append_invariance_passed
    checks["identity_bucket_amount_side_clock_reproducible"] = (
        reproducible and _clock_reproducible(primary)
    )
    for name, floor in SUPPORT_FLOORS.items():
        checks[f"{name}_accepted_minimum"] = periods[name]["accepted"] >= floor
    for name in ("selection", "future25", "future26"):
        diagnostic = periods[name]
        checks[f"{name}_maximum_month_share_at_most_half"] = bool(
            diagnostic["accepted"]
            and diagnostic["maximum_month_count"] * 2 <= diagnostic["accepted"]
        )
    checks["maximum_full_entry_gap_at_most_240_days"] = bool(
        len(primary) >= 2 and max_gap is not None and max_gap <= 240 * 86400
    )
    audit = {
        "period_diagnostics": periods,
        "maximum_full_entry_gap_seconds": max_gap,
        "gap_definition": (
            "consecutive accepted primary full entries; boundary gaps excluded"
        ),
        "long_short_counts_are_report_only": True,
        "source_gate_values": {
            name: int(audit_values.get(name, 0)) for name in source_gate_names
        },
        "manifest_source_integrity": dict(manifest_integrity),
    }
    return audit, checks


def delayed_control_boundary_crossers(
    primary: pd.DataFrame, delayed: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    if (
        len(primary) != len(delayed)
        or primary["source_identity"].tolist() != delayed["source_identity"].tolist()
    ):
        raise RuntimeError("TUSI-168 delayed control parent membership drift")
    report: dict[str, dict[str, Any]] = {}
    for name, (start, end) in {
        **SPLITS,
        **DIAGNOSTIC_PERIODS,
    }.items():
        parent_ids = set(
            project_clock_to_period(primary, start, end)["source_identity"]
        )
        shifted_ids = set(
            project_clock_to_period(delayed, start, end)["source_identity"]
        )
        identities = sorted(parent_ids - shifted_ids)
        report[name] = {
            "count": len(identities),
            "source_identities": identities,
        }
    return report


def validate_registration_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("policy_id") != POLICY_ID:
        raise RuntimeError("TUSI-168 preregistration policy drift")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("TUSI-168 preregistration canonical hash drift")
    feature = payload.get("feature_and_signal", {})
    execution = payload.get("execution", {})
    if (
        feature.get("eligible_event_types") != ["Issue", "Redeem"]
        or execution.get("hold_hours") != 168
        or payload.get("source", {}).get("deprecate_terminates_source_v1") is not True
    ):
        raise RuntimeError("TUSI-168 preregistration frozen contract drift")
    return payload


def validate_preregistration(
    path: str | Path = DEFAULT_PREREGISTRATION,
    *,
    registration: Mapping[str, Any] | None = None,
    production: bool = False,
) -> Mapping[str, Any]:
    if registration is not None:
        return validate_registration_payload(registration)
    candidate = _path(path)
    if not candidate.exists():
        if production:
            raise RuntimeError("TUSI-168 committed preregistration is missing")
        return {"policy_id": POLICY_ID, "mode": "synthetic_unregistered"}
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError("TUSI-168 preregistration path is unsafe")
    if production:
        if PREREGISTRATION_SHA256 is None:
            raise RuntimeError(
                "TUSI-168 production preregistration SHA-256 is not bound"
            )
        relative = candidate.relative_to(REPOSITORY_ROOT)
        _git("ls-files", "--error-unmatch", "--", relative.as_posix())
        _git("diff", "--quiet", "HEAD", "--", relative.as_posix())
        committed = _git("show", f"HEAD:{relative.as_posix()}")
        committed_sha256 = hashlib.sha256(committed).hexdigest()
        if (
            committed_sha256 != PREREGISTRATION_SHA256
            or sha256_file(candidate) != PREREGISTRATION_SHA256
        ):
            raise RuntimeError("TUSI-168 committed preregistration hash drift")
    raw = candidate.read_bytes()
    payload = json.loads(raw)
    if raw != prereg.canonical_manifest_bytes(payload):
        raise RuntimeError("TUSI-168 preregistration serialization drift")
    prereg.validate_manifest(payload)
    if (
        production
        and payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
    ):
        raise RuntimeError(
            "TUSI-168 committed preregistration manifest hash drift"
        )
    return validate_registration_payload(payload)


def _manifest_source_audit(
    payload: Mapping[str, Any], frame: pd.DataFrame
) -> dict[str, Any]:
    counts = {
        name: int(frame["event_type"].eq(name).sum()) for name in EVENT_DIRECTIONS
    }
    if payload.get("event_counts") != counts:
        raise RuntimeError("TUSI-168 source manifest event counts drift")
    if payload.get("event_count") != len(frame):
        raise RuntimeError("TUSI-168 source manifest row count drift")
    category_counts = payload.get("category_counts")
    if not isinstance(category_counts, Mapping) or set(category_counts) != set(
        source_builder.CATEGORIES
    ):
        raise RuntimeError("TUSI-168 source category-count schema drift")
    if (
        category_counts[source_builder.CATEGORY_SEMANTIC] != len(frame)
        or category_counts[source_builder.CATEGORY_MINT] != counts["Issue"]
        or category_counts[source_builder.CATEGORY_BURN] != counts["Redeem"]
    ):
        raise RuntimeError("TUSI-168 source category/event count drift")
    hashes = payload.get("category_canonical_sha256")
    if (
        not isinstance(hashes, Mapping)
        or set(hashes) != set(source_builder.CATEGORIES)
        or not all(SHA256.fullmatch(str(value)) for value in hashes.values())
        or not SHA256.fullmatch(str(payload.get("global_canonical_sha256")))
    ):
        raise RuntimeError("TUSI-168 source replay hash schema drift")
    if payload.get("global_log_count") != sum(
        _integer(value, "source category count")
        for value in category_counts.values()
    ):
        raise RuntimeError("TUSI-168 global log count drift")
    source_integrity = payload.get("source_integrity")
    if source_integrity != source_builder.ZERO_SOURCE_INTEGRITY:
        raise RuntimeError("TUSI-168 source integrity is not exact zero")
    source_integrity = cast(Mapping[str, int], source_integrity)
    guards = payload.get("protocol_guards")
    if guards != {
        "retry_backoff_fallback_resume": False,
        "response_dependent_sleep": False,
        "deprecate_terminal": True,
        "market_policy_performance_opened": False,
    }:
        raise RuntimeError("TUSI-168 source protocol guards drift")
    return {
        "dual_raw_log_replay_differences": 0,
        "chunk_integrity_differences": 0,
        "receipt_header_differences": 0,
        "issue_mint_transfer_pair_differences": 0,
        "redeem_burn_transfer_pair_differences": 0,
        "deprecate_events": 0,
        "source_integrity": dict(source_integrity),
    }


def _source_manifest_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in _frame_records(frame):
        paired_transfer_log_index = row["paired_transfer_log_index"]
        records.append(
            {
                "event_type": str(row["event_type"]),
                "supply_direction": int(row["supply_direction"]),
                "actor_address": str(row["actor_address"]),
                "amount_raw": str(int(row["amount_raw"])),
                "block_number": int(row["block_number"]),
                "block_hash": str(row["block_hash"]),
                "transaction_hash": str(row["transaction_hash"]),
                "transaction_index": int(row["transaction_index"]),
                "log_index": int(row["log_index"]),
                "paired_transfer_log_index": (
                    ""
                    if paired_transfer_log_index is None
                    or bool(pd.isna(paired_transfer_log_index))
                    else int(paired_transfer_log_index)
                ),
                "event_timestamp_utc": _format_time(
                    row["event_timestamp_utc"]
                ),
                "confirmation_block": int(row["confirmation_block"]),
                "confirmation_block_hash": str(
                    row["confirmation_block_hash"]
                ),
                "available_at_utc": _format_time(row["available_at_utc"]),
            }
        )
    return records


def _validate_source_generation_commit(
    payload: Any, *, production: bool
) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("TUSI-168 source manifest is not an object")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("TUSI-168 source manifest canonical hash drift")
    expected = (
        source_builder.PRODUCTION_GENERATION_COMMIT
        if production
        else source_builder.SYNTHETIC_GENERATION_COMMIT
    )
    if payload.get("generation_commit") != expected:
        raise RuntimeError("TUSI-168 source generation commit-marker drift")
    if production and (
        expected.get("manifest_is_commit_marker") is not True
        or expected.get("canonical_publication_eligible") is not True
        or expected.get("full_envelope_integrity") is not True
    ):
        raise RuntimeError("TUSI-168 builder production commit-marker drift")
    if not SHA256.fullmatch(str(payload.get("source_csv_sha256"))):
        raise RuntimeError("TUSI-168 source generation CSV binding drift")


def _validate_boundary_evidence(value: Any, *, production: bool) -> None:
    boundary = _require_exact_keys(
        value,
        {
            "outside_before_count",
            "outside_after_maximum_admissible_count",
            "header_count",
            "canonical_header_set_sha256",
            "frozen_header_set_exact",
            "boundaries",
        },
        "source boundary evidence",
    )
    frozen = boundary["frozen_header_set_exact"]
    if (
        boundary["outside_before_count"] != 0
        or boundary["outside_after_maximum_admissible_count"] != 0
        or boundary["header_count"] != 2 * len(source_builder.FROZEN_BOUNDARIES)
        or not SHA256.fullmatch(str(boundary["canonical_header_set_sha256"]))
        or not isinstance(frozen, bool)
        or not isinstance(boundary["boundaries"], list)
        or len(boundary["boundaries"]) != len(source_builder.FROZEN_BOUNDARIES)
    ):
        raise RuntimeError("TUSI-168 boundary evidence value drift")
    if (production or frozen) and (
        frozen is not True
        or boundary["canonical_header_set_sha256"]
        != source_builder.BOUNDARY_HEADER_SET_SHA256
    ):
        raise RuntimeError("TUSI-168 frozen boundary evidence drift")
    expected_keys = {
        "utc",
        "previous_block",
        "first_block_at_or_after",
        "parent_relation_exact",
        "timestamp_relation_exact",
        "frozen_hash_exact",
    }
    for observed, expected in zip(
        boundary["boundaries"],
        source_builder.FROZEN_BOUNDARIES,
        strict=True,
    ):
        expected_number = expected["number"]
        if isinstance(expected_number, bool) or not isinstance(
            expected_number, int
        ):
            raise RuntimeError("TUSI-168 frozen boundary number drift")
        item = _require_exact_keys(
            observed, expected_keys, "source boundary row"
        )
        if item != {
            "utc": expected["utc"],
            "previous_block": expected_number - 1,
            "first_block_at_or_after": expected_number,
            "parent_relation_exact": True,
            "timestamp_relation_exact": True,
            "frozen_hash_exact": frozen,
        }:
            raise RuntimeError("TUSI-168 boundary row drift")


def validate_source_manifest(
    payload: Mapping[str, Any],
    *,
    csv_bytes: bytes,
    frame: pd.DataFrame,
    production: bool = False,
) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        {
            "protocol_version",
            "source_only",
            "protocol_parent_commit",
            "replay_claim_commit",
            "replay_claim_sha256",
            "generation_commit",
            "chain",
            "source_range",
            "transports",
            "source_replay_schedule",
            "transport_exact_set_equal",
            "category_counts",
            "category_canonical_sha256",
            "global_log_count",
            "global_canonical_sha256",
            "event_counts",
            "event_count",
            "event_canonical_sha256",
            "year_counts",
            "source_csv_sha256",
            "receipt_count",
            "receipt_canonical_sha256",
            "header_count",
            "header_canonical_sha256",
            "common_finalized_head",
            "boundary_evidence",
            "protocol_guards",
            "outcome_access",
            "source_integrity",
            "manifest_hash",
        },
        "source manifest",
    )
    _validate_source_generation_commit(payload, production=production)
    if production:
        try:
            source_builder._validate_production_manifest(payload, csv_bytes)
        except source_builder.TerminalSourceFailure as exc:
            raise RuntimeError(
                "TUSI-168 builder source-manifest validation failed"
            ) from exc
    if (
        payload.get("protocol_version") != source_builder.PROTOCOL_VERSION
        or payload.get("source_only") is not True
        or payload.get("transport_exact_set_equal") is not True
        or payload.get("source_csv_sha256")
        != hashlib.sha256(csv_bytes).hexdigest()
    ):
        raise RuntimeError("TUSI-168 source manifest binding drift")
    direct = (
        payload["protocol_parent_commit"],
        payload["replay_claim_commit"],
        payload["replay_claim_sha256"],
    )
    if (
        direct[0] is not None
        and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", str(direct[0]))
        is None
    ) or (
        (direct[1] is None) != (direct[2] is None)
        or direct[1] is not None
        and (
            direct[0] is None
            or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", str(direct[1]))
            is None
            or not SHA256.fullmatch(str(direct[2]))
        )
    ):
        raise RuntimeError("TUSI-168 source direct claim/seal binding drift")
    if production and any(value is None for value in direct):
        raise RuntimeError("TUSI-168 production source claim/seal binding missing")
    chain = payload.get("chain")
    if chain != {
        "name": "TRON mainnet",
        "chain_id": source_builder.CHAIN_ID_HEX,
        "usdt_contract_base58": source_builder.USDT_CONTRACT_BASE58,
        "usdt_contract_evm": source_builder.USDT_CONTRACT,
    }:
        raise RuntimeError("TUSI-168 source chain authority drift")
    source_range = _require_exact_keys(
        payload.get("source_range"),
        {
            "start_block_inclusive",
            "start_utc",
            "end_block_exclusive",
            "last_source_block",
            "block_count",
            "chunk_size_inclusive",
            "chunk_count",
            "full_chunk_count",
            "final_chunk",
            "confirmation_blocks",
            "maximum_admissible_event_block",
            "last_confirmation_block",
            "causal_end_boundary",
        },
        "source range",
    )
    expected_range = source_builder._source_range_manifest(
        source_builder.frozen_chunks()
    )
    if production and source_range != expected_range:
        raise RuntimeError("TUSI-168 production source range drift")
    final_chunk = _require_exact_keys(
        source_range["final_chunk"],
        {"first_block", "last_block", "block_count"},
        "source final chunk",
    )
    integer_range_keys = (
        "start_block_inclusive",
        "end_block_exclusive",
        "last_source_block",
        "block_count",
        "chunk_size_inclusive",
        "chunk_count",
        "full_chunk_count",
        "confirmation_blocks",
        "maximum_admissible_event_block",
        "last_confirmation_block",
    )
    if any(
        isinstance(source_range[key], bool)
        or not isinstance(source_range[key], int)
        for key in integer_range_keys
    ) or any(
        isinstance(final_chunk[key], bool)
        or not isinstance(final_chunk[key], int)
        for key in final_chunk
    ):
        raise RuntimeError("TUSI-168 source range integer drift")
    causal_boundary = _require_exact_keys(
        source_range["causal_end_boundary"],
        {"block", "utc"},
        "source causal boundary",
    )
    first_block = source_range["start_block_inclusive"]
    last_block = source_range["last_source_block"]
    if (
        first_block < source_builder.SOURCE_START_BLOCK
        or last_block > source_builder.LAST_EVENT_BLOCK
        or source_range["end_block_exclusive"] != last_block + 1
        or source_range["block_count"] != last_block - first_block + 1
        or source_range["chunk_size_inclusive"] != source_builder.CHUNK_SIZE
        or source_range["chunk_count"] <= 0
        or not 0 <= source_range["full_chunk_count"] <= source_range["chunk_count"]
        or source_range["start_utc"]
        != (
            source_builder.SOURCE_START_UTC
            if first_block == source_builder.SOURCE_START_BLOCK
            else None
        )
        or final_chunk["first_block"] < first_block
        or final_chunk["last_block"] != last_block
        or final_chunk["block_count"]
        != final_chunk["last_block"] - final_chunk["first_block"] + 1
        or not 1 <= final_chunk["block_count"] <= source_builder.CHUNK_SIZE
        or source_range["confirmation_blocks"]
        != source_builder.CONFIRMATION_BLOCKS
        or source_range["maximum_admissible_event_block"]
        != source_builder.LAST_EVENT_BLOCK
        or source_range["last_confirmation_block"]
        != source_builder.LAST_CONFIRMATION_BLOCK
        or causal_boundary
        != {
            "block": source_builder.END_BOUNDARY_BLOCK,
            "utc": source_builder.END_BOUNDARY_UTC,
        }
        or bool(_series(frame, "block_number").lt(first_block).any())
        or bool(_series(frame, "block_number").gt(last_block).any())
    ):
        raise RuntimeError("TUSI-168 source range drift")
    expected_transports = [
        dict(identity)
        for identity in source_builder.validate_transport_identities(
            source_builder.SANITIZED_TRANSPORTS
        )
    ]
    if payload.get("transports") != expected_transports:
        raise RuntimeError("TUSI-168 source transport schema drift")
    if payload.get("source_replay_schedule") != {
        "inter_batch_throttle_seconds": source_builder.PRODUCTION_THROTTLE_SECONDS,
        "maximum_batch_by_role": dict(source_builder.TRANSPORT_MAX_BATCH),
        "rpc_methods": sorted(source_builder.RPC_METHODS),
    }:
        raise RuntimeError("TUSI-168 source replay schedule drift")

    records = _source_manifest_records(frame)
    if csv_bytes != source_builder.serialize_csv(records):
        raise RuntimeError("TUSI-168 source CSV canonical row bytes drift")
    if payload.get("event_canonical_sha256") != canonical_hash(records):
        raise RuntimeError("TUSI-168 canonical source event hash drift")
    year_counts = {str(year): 0 for year in range(2023, 2027)}
    for record in records:
        year_counts[record["event_timestamp_utc"][:4]] += 1
    if payload.get("year_counts") != year_counts:
        raise RuntimeError("TUSI-168 source year counts drift")
    for count_name in ("receipt_count", "header_count"):
        value = payload.get(count_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("TUSI-168 source evidence count drift")
    if payload["receipt_count"] != frame["transaction_hash"].nunique():
        raise RuntimeError("TUSI-168 source receipt count drift")
    for hash_name in (
        "receipt_canonical_sha256",
        "header_canonical_sha256",
    ):
        if not SHA256.fullmatch(str(payload.get(hash_name))):
            raise RuntimeError("TUSI-168 source evidence hash drift")

    finalized = _require_exact_keys(
        payload.get("common_finalized_head"),
        {"number", "hash", "timestamp_utc", "covers_last_confirmation"},
        "source finalized head",
    )
    finalized_number = finalized["number"]
    if (
        isinstance(finalized_number, bool)
        or not isinstance(finalized_number, int)
        or finalized_number < 0
        or not HEX_64.fullmatch(str(finalized.get("hash")))
        or not isinstance(finalized.get("timestamp_utc"), str)
        or _format_time(finalized.get("timestamp_utc"))
        != str(finalized.get("timestamp_utc"))
        or finalized.get("covers_last_confirmation") is not True
        or (
            production
            and finalized_number < source_builder.LAST_CONFIRMATION_BLOCK
        )
    ):
        raise RuntimeError("TUSI-168 finalized-head coverage drift")
    _validate_boundary_evidence(
        payload.get("boundary_evidence"), production=production
    )
    if payload.get("outcome_access") != {
        "btc_market_rows_opened": 0,
        "funding_rows_opened": 0,
        "returns_opened": 0,
        "pnl_opened": 0,
        "cagr_opened": 0,
        "strict_mdd_opened": 0,
        "outcomes_opened": 0,
    }:
        raise RuntimeError("TUSI-168 source outcome-access boundary drift")
    if production:
        _validate_production_claim_binding(payload)
    return _manifest_source_audit(payload, frame)


def _validate_production_claim_binding(manifest: Mapping[str, Any]) -> None:
    claim_path = _path(source_builder.REPLAY_CLAIM_PATH)
    if claim_path.is_symlink() or not claim_path.is_file():
        raise RuntimeError("TUSI-168 replay claim is absent or unsafe")
    relative = source_builder.REPLAY_CLAIM_PATH.as_posix()
    _git("ls-files", "--error-unmatch", "--", relative)
    _git("diff", "--quiet", "HEAD", "--", relative)
    raw = claim_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["replay_claim_sha256"]:
        raise RuntimeError("TUSI-168 replay claim file hash drift")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise RuntimeError("TUSI-168 replay claim payload drift")
    protocol_seal = payload.get("protocol_seal")
    if not isinstance(protocol_seal, Mapping):
        raise RuntimeError("TUSI-168 replay claim protocol seal missing")
    expected = source_builder._claim_payload(
        protocol_seal, source_builder.SANITIZED_TRANSPORTS
    )
    if (
        payload != expected
        or raw != source_builder._canonical_json_bytes(payload, trailing_lf=True)
    ):
        raise RuntimeError("TUSI-168 replay claim payload drift")
    claim_commit = str(manifest["replay_claim_commit"])
    protocol_parent = str(manifest["protocol_parent_commit"])
    if (
        not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", claim_commit)
        or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", protocol_parent)
        or protocol_parent != protocol_seal.get("git_head")
    ):
        raise RuntimeError("TUSI-168 replay claim commit binding drift")
    source_builder.validate_protocol_seal(protocol_seal)
    parent_line = (
        _git("rev-list", "--parents", "-n", "1", claim_commit)
        .decode("ascii")
        .strip()
        .split()
    )
    if parent_line != [claim_commit, protocol_parent]:
        raise RuntimeError("TUSI-168 replay claim parent relation drift")
    if _git("show", f"{claim_commit}:{relative}") != raw:
        raise RuntimeError("TUSI-168 replay claim committed bytes drift")


def _decode_csv_bytes(raw: bytes) -> pd.DataFrame:
    try:
        decoded = gzip.decompress(raw)
    except (OSError, EOFError):
        raise RuntimeError("TUSI-168 source CSV is not valid gzip") from None
    with io.StringIO(decoded.decode("utf-8"), newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError("TUSI-168 source CSV is empty") from None
        rows = list(reader)
    if header != list(SOURCE_COLUMNS):
        _reject_forbidden_columns(header)
        raise RuntimeError("TUSI-168 source CSV header drift")
    if any(len(row) != len(header) for row in rows):
        raise RuntimeError("TUSI-168 source CSV row width drift")
    return pd.read_csv(
        io.BytesIO(decoded),
        dtype={
            "amount_raw": "string",
            "block_number": "string",
            "transaction_index": "string",
            "log_index": "string",
            "paired_transfer_log_index": "string",
            "confirmation_block": "string",
        },
        keep_default_na=True,
    )


def load_source_artifacts(
    *,
    csv_path: str | Path,
    manifest_path: str | Path,
    production: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_candidate = _path(csv_path)
    manifest_candidate = _path(manifest_path)
    for candidate in (csv_candidate, manifest_candidate):
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError("TUSI-168 source artifact path is unsafe")
    if production:
        relative = manifest_candidate.relative_to(REPOSITORY_ROOT).as_posix()
        _git("ls-files", "--error-unmatch", "--", relative)
        _git("diff", "--quiet", "HEAD", "--", relative)
    manifest_bytes = manifest_candidate.read_bytes()
    try:
        payload = source_builder._decode_canonical_manifest(manifest_bytes)
    except source_builder.TerminalSourceFailure as exc:
        raise RuntimeError(
            "TUSI-168 source manifest serialization drift"
        ) from exc
    _validate_source_generation_commit(payload, production=production)
    if production:
        relative = csv_candidate.relative_to(REPOSITORY_ROOT).as_posix()
        _git("ls-files", "--error-unmatch", "--", relative)
        _git("diff", "--quiet", "HEAD", "--", relative)
    csv_bytes = csv_candidate.read_bytes()
    if hashlib.sha256(csv_bytes).hexdigest() != payload["source_csv_sha256"]:
        raise RuntimeError("TUSI-168 source generation CSV binding drift")
    raw_frame = _decode_csv_bytes(csv_bytes)
    frame = validate_source_frame(raw_frame)
    gates = validate_source_manifest(
        payload,
        csv_bytes=csv_bytes,
        frame=frame,
        production=production,
    )
    return raw_frame, {
        "artifact_eligible": production,
        "source_csv_path": str(csv_path),
        "source_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "source_csv_bytes": len(csv_bytes),
        "source_rows": len(frame),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_manifest_hash": payload["manifest_hash"],
        **gates,
    }


def load_source_manifest() -> tuple[pd.DataFrame, dict[str, Any]]:
    return load_source_artifacts(
        csv_path=DEFAULT_SOURCE_CSV,
        manifest_path=DEFAULT_SOURCE_MANIFEST,
        production=True,
    )


def load_synthetic_source_artifacts(
    *,
    csv_path: str | Path,
    manifest_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return load_source_artifacts(
        csv_path=csv_path,
        manifest_path=manifest_path,
        production=False,
    )


def _build_support_from_frame(
    frame: pd.DataFrame,
    *,
    registration: Mapping[str, Any] | None = None,
    source_audit: Mapping[str, Any] | None = None,
    artifact_eligible: bool,
) -> tuple[dict[str, Any], bytes, bytes]:
    if not isinstance(artifact_eligible, bool):
        raise RuntimeError("TUSI-168 artifact eligibility type drift")
    source_claims_artifact = (
        (source_audit or {}).get("artifact_eligible") is True
    )
    if source_claims_artifact is not artifact_eligible:
        raise RuntimeError("TUSI-168 source artifact eligibility/audit drift")
    validated = validate_source_frame(frame)
    registration_payload = validate_preregistration(registration=registration)
    controls, raw_counts = _build_controls_from_validated(validated)
    reproduced, reproduced_counts = _build_controls_from_validated(validated.copy())
    reproducible = raw_counts == reproduced_counts and all(
        _csv_bytes(controls[name], compress=False)
        == _csv_bytes(reproduced[name], compress=False)
        for name in CONTROL_ORDER
    )
    append_passed, append_report = future_append_selection_invariance(frame)
    audit, checks = support_checks(
        controls["primary"],
        source_audit=source_audit,
        append_invariance_passed=append_passed,
        reproducible=reproducible,
    )
    passed = all(checks.values())
    primary_bytes = _csv_bytes(controls["primary"])
    control_parts = [
        controls[name] for name in CONTROL_ORDER[1:] if not controls[name].empty
    ]
    control_frame = (
        pd.concat(control_parts, ignore_index=True) if control_parts else _empty_clock()
    )
    control_bytes = _csv_bytes(control_frame)
    source_values = dict(source_audit or {})
    unknown_source_audit = set(source_values) - {
        *SOURCE_CONTRACT_AUDIT_KEYS,
        "source_rows",
    }
    if unknown_source_audit:
        raise RuntimeError("TUSI-168 source audit schema drift")
    source_contract_values = {
        "artifact_eligible": bool(source_values.get("artifact_eligible", False)),
        "source_csv_path": source_values.get(
            "source_csv_path", "synthetic_injected_frame"
        ),
        "source_csv_sha256": source_values.get("source_csv_sha256"),
        "source_csv_bytes": source_values.get("source_csv_bytes"),
        "source_manifest_path": source_values.get(
            "source_manifest_path", "synthetic_injected_manifest"
        ),
        "source_manifest_sha256": source_values.get("source_manifest_sha256"),
        "source_manifest_hash": source_values.get("source_manifest_hash"),
        "dual_raw_log_replay_differences": int(
            source_values.get("dual_raw_log_replay_differences", 0)
        ),
        "chunk_integrity_differences": int(
            source_values.get("chunk_integrity_differences", 0)
        ),
        "receipt_header_differences": int(
            source_values.get("receipt_header_differences", 0)
        ),
        "issue_mint_transfer_pair_differences": int(
            source_values.get("issue_mint_transfer_pair_differences", 0)
        ),
        "redeem_burn_transfer_pair_differences": int(
            source_values.get("redeem_burn_transfer_pair_differences", 0)
        ),
        "deprecate_events": int(source_values.get("deprecate_events", 0)),
        "source_integrity": dict(
            source_values.get("source_integrity", source_builder.ZERO_SOURCE_INTEGRITY)
        ),
    }
    delayed_crossers = delayed_control_boundary_crossers(
        controls["primary"], controls["one_bar_delayed_entry"]
    )
    support_audit = {
        key: value for key, value in audit.items() if key != "period_diagnostics"
    }
    support_audit["one_bar_delayed_entry_boundary_crossers"] = delayed_crossers
    evidence = dict(EVIDENCE_BOUNDARY)
    evidence["source_rows_opened"] = len(validated)
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "status": (
            "source_support_passed"
            if artifact_eligible and passed
            else "retired_before_novelty"
            if artifact_eligible
            else "synthetic_only_nonpublishable"
        ),
        "terminal": artifact_eligible,
        "artifact_eligible": artifact_eligible,
        "support_passed": passed,
        "decision": (
            "SOURCE_SUPPORT_PASS"
            if artifact_eligible and passed
            else "RETIRE_TUSI_168_UNCHANGED_BEFORE_NOVELTY"
            if artifact_eligible
            else "SYNTHETIC_ONLY_NO_SOURCE_SUPPORT_DECISION"
        ),
        "registration": {
            "manifest_hash": registration_payload.get("manifest_hash"),
            "mode": (
                "artifact"
                if artifact_eligible
                else "injected"
                if registration is not None
                else registration_payload.get("mode", "artifact")
            ),
        },
        "source_contract": {
            "columns": list(SOURCE_COLUMNS),
            "rows": len(validated),
            **source_contract_values,
        },
        "raw_candidate_counts": raw_counts,
        "accepted_clock_counts": {name: len(controls[name]) for name in CONTROL_ORDER},
        "period_diagnostics": {
            name: audit["period_diagnostics"][name] for name in PERIOD_ORDER
        },
        "support_audit": support_audit,
        "support_checks": checks,
        "future_append_selection_invariance": append_report,
        "control_overlap": control_overlap(controls),
        "clock_artifacts": {
            "primary_sha256": hashlib.sha256(primary_bytes).hexdigest(),
            "controls_sha256": hashlib.sha256(control_bytes).hexdigest(),
        },
        "evidence_boundary": evidence,
        "source_support_precedes_novelty": True,
        "novelty_comparator_market_or_outcome_artifacts_opened": False,
    }
    return {**core, "manifest_hash": canonical_hash(core)}, primary_bytes, control_bytes


def build_support_from_frame(
    frame: pd.DataFrame,
    *,
    registration: Mapping[str, Any] | None = None,
    source_audit: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Build a synthetic, explicitly nonpublishable support generation."""

    if (source_audit or {}).get("artifact_eligible") is True:
        raise RuntimeError(
            "TUSI-168 synthetic source audit may not claim artifact eligibility"
        )
    return _build_support_from_frame(
        frame,
        registration=registration,
        source_audit=source_audit,
        artifact_eligible=False,
    )


def _git(*arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"TUSI-168 git validation failed: {' '.join(arguments)}")
    return completed.stdout


def _assert_evaluator_committed() -> None:
    paths = (SCRIPT_PATH.as_posix(), TEST_PATH.as_posix())
    _git("ls-files", "--error-unmatch", "--", *paths)
    _git("diff", "--quiet", "HEAD", "--", *paths)


class _OutputTarget(NamedTuple):
    path: Path
    parent: Path
    leaf: str


def _normalize_output_target(path: str | Path) -> _OutputTarget:
    candidate = Path(path)
    if ".." in candidate.parts:
        raise RuntimeError("TUSI-168 output parent traversal alias is forbidden")
    absolute = candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate
    normalized_text = os.path.normpath(os.fspath(absolute))
    if normalized_text.startswith(os.sep * 2):
        normalized_text = os.sep + normalized_text.lstrip(os.sep)
    normalized = Path(normalized_text)
    if not normalized.is_absolute() or normalized.name in {"", ".", ".."}:
        raise RuntimeError("TUSI-168 output target is invalid")
    return _OutputTarget(normalized, normalized.parent, normalized.name)


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise RuntimeError("TUSI-168 secure directory traversal is unavailable")
    return (
        os.O_RDONLY
        | nofollow
        | directory
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_output_parent(parent: Path, *, create: bool) -> int | None:
    """Open an absolute parent without following any component symlink."""

    if not parent.is_absolute():
        raise RuntimeError("TUSI-168 output parent must be absolute")
    flags = _directory_flags()
    descriptor = os.open(parent.anchor, flags)
    try:
        for component in parent.parts[1:]:
            try:
                child = os.open(
                    component,
                    flags,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    os.close(descriptor)
                    return None
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    os.fsync(descriptor)
                try:
                    child = os.open(
                        component,
                        flags,
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    raise RuntimeError(
                        "TUSI-168 output ancestor is unsafe"
                    ) from exc
            except OSError as exc:
                raise RuntimeError("TUSI-168 output ancestor is unsafe") from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _read_output_at(parent_fd: int, leaf: str) -> bytes | None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(leaf, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError("TUSI-168 output leaf is unsafe") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("TUSI-168 output leaf is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _read_existing_output(target: _OutputTarget) -> bytes | None:
    parent_fd = _open_output_parent(target.parent, create=False)
    if parent_fd is None:
        return None
    try:
        return _read_output_at(parent_fd, target.leaf)
    finally:
        os.close(parent_fd)


def _check_output_target(target: _OutputTarget) -> None:
    parent_fd = _open_output_parent(target.parent, create=False)
    if parent_fd is None:
        return
    try:
        try:
            leaf_stat = os.stat(
                target.leaf,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if not stat.S_ISREG(leaf_stat.st_mode):
            raise RuntimeError("TUSI-168 output leaf is unsafe")
    finally:
        os.close(parent_fd)


def _stage(parent_fd: int, output_leaf: str, payload: bytes) -> str:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(128):
        temporary = f".{output_leaf}.{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                temporary,
                flags,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            continue
        break
    else:
        raise RuntimeError("TUSI-168 could not allocate a staged output")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise
    return temporary


def _fsync_directory(directory: Path, *, directory_fd: int) -> None:
    if not directory.is_absolute():
        raise RuntimeError("TUSI-168 fsync directory must be absolute")
    os.fsync(directory_fd)


def _same_inode(parent_fd: int, left: str, right: str) -> bool:
    try:
        left_stat = os.stat(left, dir_fd=parent_fd, follow_symlinks=False)
        right_stat = os.stat(right, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (
        stat.S_ISREG(left_stat.st_mode)
        and stat.S_ISREG(right_stat.st_mode)
        and (left_stat.st_dev, left_stat.st_ino)
        == (right_stat.st_dev, right_stat.st_ino)
    )


def _require_exact_keys(
    value: Any, expected: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError(f"TUSI-168 {label} exact schema drift")
    return value


def _validate_clock_csv(
    payload: bytes, *, primary: bool
) -> tuple[list[dict[str, str]], dict[str, int]]:
    try:
        raw = gzip.decompress(payload)
        text = raw.decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError):
        raise RuntimeError("TUSI-168 support clock gzip/UTF-8 drift") from None
    if payload != _deterministic_gzip(raw) or not text.endswith("\n"):
        raise RuntimeError("TUSI-168 support clock serialization drift")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != list(CLOCK_COLUMNS):
        raise RuntimeError("TUSI-168 support clock exact header drift")
    rows = list(reader)
    allowed = ("primary",) if primary else CONTROL_ORDER[1:]
    counts = {name: 0 for name in allowed}
    prior_control_index = -1
    prior_sort: dict[str, tuple[str, str, str, str]] = {}
    seen_identities: dict[str, set[str]] = {name: set() for name in allowed}
    for row in rows:
        if set(row) != set(CLOCK_COLUMNS) or any(
            value is None or value == "" for value in row.values()
        ):
            raise RuntimeError("TUSI-168 support clock null/row schema drift")
        if row["policy_id"] != POLICY_ID or row["control"] not in allowed:
            raise RuntimeError("TUSI-168 support clock policy/control drift")
        control_index = allowed.index(row["control"])
        if control_index < prior_control_index:
            raise RuntimeError("TUSI-168 support control order drift")
        prior_control_index = control_index
        if row["window"] not in SPLITS:
            raise RuntimeError("TUSI-168 support clock window drift")
        try:
            constituents = json.loads(row["constituent_identities_json"])
        except json.JSONDecodeError:
            raise RuntimeError("TUSI-168 support constituent JSON drift") from None
        canonical_constituents = json.dumps(
            constituents,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        if (
            canonical_constituents != row["constituent_identities_json"]
            or not isinstance(constituents, list)
            or constituents != sorted(constituents)
            or hashlib.sha256(canonical_constituents.encode()).hexdigest()
            != row["source_identity"]
            or not SHA256.fullmatch(row["source_identity"])
        ):
            raise RuntimeError("TUSI-168 support source identity drift")
        for identity in constituents:
            if (
                not isinstance(identity, list)
                or len(identity) != 6
                or not all(isinstance(identity[index], int) for index in (0, 1, 2, 5))
                or not HEX_64.fullmatch(str(identity[3]))
                or identity[4] not in {"Issue", "Redeem", "DestroyedBlackFunds"}
            ):
                raise RuntimeError("TUSI-168 support constituent identity schema drift")
        if (
            not re.fullmatch(r"[1-9][0-9]*", row["constituent_count"])
            or int(row["constituent_count"]) != len(constituents)
            or not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", row["bucket_amount_raw"])
            or int(row["bucket_amount_raw"]) == 0
        ):
            raise RuntimeError("TUSI-168 support clock integer drift")
        for field in (
            "decision_time_utc",
            "entry_time_utc",
            "exit_time_utc",
        ):
            if _format_time(row[field]) != row[field]:
                raise RuntimeError("TUSI-168 support clock timestamp drift")
        decision = _timestamp(row["decision_time_utc"])
        entry = _timestamp(row["entry_time_utc"])
        exit_time = _timestamp(row["exit_time_utc"])
        if decision >= entry or exit_time != entry + HOLD:
            raise RuntimeError("TUSI-168 support clock causal timing drift")
        if row["control"] != "one_bar_delayed_entry":
            start, end = SPLITS[row["window"]]
            if entry < start or exit_time > end:
                raise RuntimeError(
                    "TUSI-168 support clock main-window containment drift"
                )
        amount = int(row["bucket_amount_raw"])
        if row["control"] in {*INDEPENDENT_CONTROLS, "one_bar_delayed_entry"} and row[
            "side"
        ] != ("LONG" if amount > 0 else "SHORT"):
            raise RuntimeError("TUSI-168 support clock side drift")
        order = (
            row["entry_time_utc"],
            row["decision_time_utc"],
            row["source_identity"],
            row["side"],
        )
        if row["control"] in prior_sort and order < prior_sort[row["control"]]:
            raise RuntimeError("TUSI-168 support clock row order drift")
        if row["source_identity"] in seen_identities[row["control"]]:
            raise RuntimeError("TUSI-168 duplicate support source identity")
        seen_identities[row["control"]].add(row["source_identity"])
        prior_sort[row["control"]] = order
        counts[row["control"]] += 1
    return rows, counts


def _support_check_keys() -> set[str]:
    return {
        *{
            f"source_{name}_zero"
            for name in (
                "dual_raw_log_replay_differences",
                "chunk_integrity_differences",
                "receipt_header_differences",
                "issue_mint_transfer_pair_differences",
                "redeem_burn_transfer_pair_differences",
                "deprecate_events",
            )
        },
        "future_append_selection_differences_zero",
        "identity_bucket_amount_side_clock_reproducible",
        *{f"{name}_accepted_minimum" for name in SUPPORT_FLOORS},
        *{
            f"{name}_maximum_month_share_at_most_half"
            for name in ("selection", "future25", "future26")
        },
        "maximum_full_entry_gap_at_most_240_days",
        *{
            f"manifest_source_integrity_{name}_zero"
            for name in source_builder.ZERO_SOURCE_INTEGRITY
        },
    }


def _validate_report_schema(
    report: Mapping[str, Any],
    *,
    primary_counts: Mapping[str, int],
    control_counts: Mapping[str, int],
    artifact_eligible_authorized: bool,
) -> None:
    _require_exact_keys(report, REPORT_KEYS, "report")
    scalar_types = {
        "protocol_version": str,
        "policy_id": str,
        "status": str,
        "terminal": bool,
        "artifact_eligible": bool,
        "support_passed": bool,
        "decision": str,
        "source_support_precedes_novelty": bool,
        "novelty_comparator_market_or_outcome_artifacts_opened": bool,
        "manifest_hash": str,
    }
    if any(
        not isinstance(report[key], expected) for key, expected in scalar_types.items()
    ):
        raise RuntimeError("TUSI-168 report top-level type drift")
    if (
        report["protocol_version"] != PROTOCOL_VERSION
        or report["policy_id"] != POLICY_ID
    ):
        raise RuntimeError("TUSI-168 report identity drift")
    if report["artifact_eligible"] is not artifact_eligible_authorized:
        raise RuntimeError("TUSI-168 artifact eligibility authorization drift")
    registration = _require_exact_keys(
        report["registration"], {"manifest_hash", "mode"}, "registration"
    )
    if registration["mode"] not in {
        "artifact",
        "injected",
        "synthetic_unregistered",
    } or (
        registration["manifest_hash"] is not None
        and not SHA256.fullmatch(str(registration["manifest_hash"]))
    ):
        raise RuntimeError("TUSI-168 registration report type drift")
    source_contract = _require_exact_keys(
        report["source_contract"],
        {"columns", "rows", *SOURCE_CONTRACT_AUDIT_KEYS},
        "source contract",
    )
    if (
        source_contract["columns"] != list(SOURCE_COLUMNS)
        or not isinstance(source_contract["rows"], int)
        or source_contract["rows"] <= 0
        or not isinstance(source_contract["artifact_eligible"], bool)
        or source_contract["artifact_eligible"] != report["artifact_eligible"]
        or not isinstance(source_contract["source_integrity"], Mapping)
        or set(source_contract["source_integrity"])
        != set(source_builder.ZERO_SOURCE_INTEGRITY)
    ):
        raise RuntimeError("TUSI-168 source contract type/value drift")
    if any(
        not isinstance(source_contract[key], str)
        for key in ("source_csv_path", "source_manifest_path")
    ) or any(
        source_contract[key] is not None
        and not SHA256.fullmatch(str(source_contract[key]))
        for key in (
            "source_csv_sha256",
            "source_manifest_sha256",
            "source_manifest_hash",
        )
    ):
        raise RuntimeError("TUSI-168 source artifact binding type drift")
    if source_contract["source_csv_bytes"] is not None and (
        isinstance(source_contract["source_csv_bytes"], bool)
        or not isinstance(source_contract["source_csv_bytes"], int)
        or source_contract["source_csv_bytes"] <= 0
    ):
        raise RuntimeError("TUSI-168 source byte count type drift")
    for key in (
        "dual_raw_log_replay_differences",
        "chunk_integrity_differences",
        "receipt_header_differences",
        "issue_mint_transfer_pair_differences",
        "redeem_burn_transfer_pair_differences",
        "deprecate_events",
    ):
        if isinstance(source_contract[key], bool) or not isinstance(
            source_contract[key], int
        ):
            raise TypeError("TUSI-168 source difference type drift")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in source_contract["source_integrity"].values()
    ):
        raise RuntimeError("TUSI-168 source integrity value type drift")
    if report["artifact_eligible"] and (
        registration["mode"] != "artifact"
        or registration["manifest_hash"] is None
        or source_contract["source_csv_path"] != str(DEFAULT_SOURCE_CSV)
        or source_contract["source_manifest_path"]
        != str(DEFAULT_SOURCE_MANIFEST)
        or source_contract["source_csv_bytes"] is None
        or any(
            source_contract[key] is None
            for key in (
                "source_csv_sha256",
                "source_manifest_sha256",
                "source_manifest_hash",
            )
        )
    ):
        raise RuntimeError("TUSI-168 artifact eligibility binding drift")
    raw_counts = _require_exact_keys(
        report["raw_candidate_counts"],
        set(CONTROL_ORDER),
        "raw candidate counts",
    )
    accepted_counts = _require_exact_keys(
        report["accepted_clock_counts"],
        set(CONTROL_ORDER),
        "accepted clock counts",
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (*raw_counts.values(), *accepted_counts.values())
    ):
        raise RuntimeError("TUSI-168 report count type drift")
    observed_counts = {"primary": primary_counts["primary"], **control_counts}
    if dict(accepted_counts) != observed_counts:
        raise RuntimeError("TUSI-168 report/clock accepted counts drift")
    if any(
        accepted_counts[name] > raw_counts[name] for name in INDEPENDENT_CONTROLS
    ) or any(
        raw_counts[name] != accepted_counts["primary"]
        or accepted_counts[name] != accepted_counts["primary"]
        for name in SAME_PARENT_CONTROLS
    ):
        raise RuntimeError("TUSI-168 raw/accepted control count consistency drift")
    periods = _require_exact_keys(
        report["period_diagnostics"],
        set(PERIOD_ORDER),
        "period diagnostics",
    )
    diagnostic_keys = {
        "accepted",
        "LONG",
        "SHORT",
        "utc_entry_month_counts",
        "maximum_month_count",
        "maximum_month_share_exact",
    }
    for name, diagnostic in periods.items():
        item = _require_exact_keys(
            diagnostic, diagnostic_keys, f"{name} period diagnostic"
        )
        if (
            any(
                isinstance(item[key], bool) or not isinstance(item[key], int)
                for key in ("accepted", "LONG", "SHORT", "maximum_month_count")
            )
            or item["accepted"] != item["LONG"] + item["SHORT"]
            or not isinstance(item["utc_entry_month_counts"], Mapping)
        ):
            raise RuntimeError("TUSI-168 period diagnostic type drift")
        monthly = cast(
            Mapping[str, int], item["utc_entry_month_counts"]
        )
        if (
            any(
                not re.fullmatch(r"[0-9]{4}-(?:0[1-9]|1[0-2])", str(month))
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                for month, count in monthly.items()
            )
            or sum(monthly.values()) != item["accepted"]
            or (max(monthly.values()) if monthly else 0) != item["maximum_month_count"]
        ):
            raise RuntimeError("TUSI-168 monthly diagnostic drift")
        share = item["maximum_month_share_exact"]
        if (share is None) is not (item["accepted"] == 0):
            raise RuntimeError("TUSI-168 month share presence drift")
        if share is not None:
            fraction = _require_exact_keys(
                share, {"numerator", "denominator"}, "month share"
            )
            if (
                any(
                    isinstance(fraction[key], bool)
                    or not isinstance(fraction[key], int)
                    for key in ("numerator", "denominator")
                )
                or fraction["denominator"] <= 0
                or fraction["numerator"] != item["maximum_month_count"]
                or fraction["denominator"] != item["accepted"]
            ):
                raise RuntimeError("TUSI-168 month share report drift")
    if (
        periods["full"]["accepted"] != accepted_counts["primary"]
        or sum(periods[name]["accepted"] for name in SPLITS)
        != accepted_counts["primary"]
    ):
        raise RuntimeError("TUSI-168 full/main-period count consistency drift")
    support_audit = _require_exact_keys(
        report["support_audit"],
        {
            "maximum_full_entry_gap_seconds",
            "gap_definition",
            "long_short_counts_are_report_only",
            "source_gate_values",
            "manifest_source_integrity",
            "one_bar_delayed_entry_boundary_crossers",
        },
        "support audit",
    )
    _require_exact_keys(
        support_audit["source_gate_values"],
        {
            "dual_raw_log_replay_differences",
            "chunk_integrity_differences",
            "receipt_header_differences",
            "issue_mint_transfer_pair_differences",
            "redeem_burn_transfer_pair_differences",
            "deprecate_events",
        },
        "source gate values",
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in support_audit["source_gate_values"].values()
    ):
        raise RuntimeError("TUSI-168 source gate value type drift")
    if (
        support_audit["maximum_full_entry_gap_seconds"] is not None
        and (
            isinstance(support_audit["maximum_full_entry_gap_seconds"], bool)
            or not isinstance(support_audit["maximum_full_entry_gap_seconds"], int)
            or support_audit["maximum_full_entry_gap_seconds"] < 0
        )
    ) or (
        support_audit["gap_definition"]
        != "consecutive accepted primary full entries; boundary gaps excluded"
        or support_audit["long_short_counts_are_report_only"] is not True
    ):
        raise RuntimeError("TUSI-168 support audit type drift")
    _require_exact_keys(
        support_audit["manifest_source_integrity"],
        set(source_builder.ZERO_SOURCE_INTEGRITY),
        "manifest source integrity",
    )
    crossers = _require_exact_keys(
        support_audit["one_bar_delayed_entry_boundary_crossers"],
        set(PERIOD_ORDER),
        "delayed boundary crossers",
    )
    for name, diagnostic in crossers.items():
        item = _require_exact_keys(
            diagnostic,
            {"count", "source_identities"},
            f"{name} delayed crossers",
        )
        if (
            not isinstance(item["count"], int)
            or not isinstance(item["source_identities"], list)
            or item["count"] != len(item["source_identities"])
            or item["source_identities"] != sorted(set(item["source_identities"]))
            or not all(
                SHA256.fullmatch(str(value)) for value in item["source_identities"]
            )
        ):
            raise RuntimeError("TUSI-168 delayed crosser report drift")
    checks = _require_exact_keys(
        report["support_checks"], _support_check_keys(), "support checks"
    )
    if not all(isinstance(value, bool) for value in checks.values()):
        raise RuntimeError("TUSI-168 support check type drift")
    if report["support_passed"] is not all(checks.values()):
        raise RuntimeError("TUSI-168 support decision/check consistency drift")
    expected_status = (
        ("source_support_passed", "SOURCE_SUPPORT_PASS")
        if report["artifact_eligible"] and report["support_passed"]
        else (
            "retired_before_novelty",
            "RETIRE_TUSI_168_UNCHANGED_BEFORE_NOVELTY",
        )
        if report["artifact_eligible"]
        else (
            "synthetic_only_nonpublishable",
            "SYNTHETIC_ONLY_NO_SOURCE_SUPPORT_DECISION",
        )
    )
    if (
        report["terminal"] is not report["artifact_eligible"]
        or (report["status"], report["decision"]) != expected_status
    ):
        raise RuntimeError("TUSI-168 terminal status/decision mapping drift")
    source_gate_values = support_audit["source_gate_values"]
    for name, value in source_gate_values.items():
        if (
            value != source_contract[name]
            or checks[f"source_{name}_zero"] is not (value == 0)
        ):
            raise RuntimeError("TUSI-168 source gate/check consistency drift")
    if (
        support_audit["manifest_source_integrity"]
        != source_contract["source_integrity"]
    ):
        raise RuntimeError("TUSI-168 source-integrity report consistency drift")
    for name, value in source_contract["source_integrity"].items():
        if checks[f"manifest_source_integrity_{name}_zero"] is not (value == 0):
            raise RuntimeError(
                "TUSI-168 source-integrity/check consistency drift"
            )
    if report["artifact_eligible"] and (
        any(value != 0 for value in source_gate_values.values())
        or source_contract["source_integrity"]
        != source_builder.ZERO_SOURCE_INTEGRITY
    ):
        raise RuntimeError("TUSI-168 artifact source-integrity eligibility drift")
    for name, floor in SUPPORT_FLOORS.items():
        if checks[f"{name}_accepted_minimum"] is not (
            periods[name]["accepted"] >= floor
        ):
            raise RuntimeError("TUSI-168 support-floor consistency drift")
    for name in ("selection", "future25", "future26"):
        diagnostic = periods[name]
        expected_month_share = bool(
            diagnostic["accepted"]
            and diagnostic["maximum_month_count"] * 2
            <= diagnostic["accepted"]
        )
        if (
            checks[f"{name}_maximum_month_share_at_most_half"]
            is not expected_month_share
        ):
            raise RuntimeError("TUSI-168 month-share/check consistency drift")
    append = _require_exact_keys(
        report["future_append_selection_invariance"],
        {"prefix_source_rule", "total_differences", "comparisons"},
        "future append",
    )
    if (
        append["prefix_source_rule"] != "available_at_utc < 2025-01-01T00:00:00Z"
        or isinstance(append["total_differences"], bool)
        or not isinstance(append["total_differences"], int)
        or append["total_differences"] < 0
    ):
        raise RuntimeError("TUSI-168 future append summary drift")
    comparisons = _require_exact_keys(
        append["comparisons"], set(CONTROL_ORDER), "future append controls"
    )
    comparison_keys = {
        "differences",
        "full_rows",
        "prefix_rows",
        "full_sha256",
        "prefix_sha256",
    }
    compared_differences = 0
    for control, views in comparisons.items():
        expected_views = (
            {"raw", "accepted"} if control in INDEPENDENT_CONTROLS else {"accepted"}
        )
        for view, comparison in _require_exact_keys(
            views, expected_views, f"{control} future append views"
        ).items():
            item = _require_exact_keys(
                comparison,
                comparison_keys,
                f"{control} {view} future append comparison",
            )
            if (
                any(
                    isinstance(item[key], bool)
                    or not isinstance(item[key], int)
                    or item[key] < 0
                    for key in ("differences", "full_rows", "prefix_rows")
                )
                or not SHA256.fullmatch(str(item["full_sha256"]))
                or not SHA256.fullmatch(str(item["prefix_sha256"]))
            ):
                raise RuntimeError("TUSI-168 future append comparison drift")
            equal_summary = (
                item["full_rows"] == item["prefix_rows"]
                and item["full_sha256"] == item["prefix_sha256"]
            )
            if (
                item["differences"] not in {0, 1}
                or (item["differences"] == 0) is not equal_summary
            ):
                raise RuntimeError(
                    "TUSI-168 future append row/hash consistency drift"
                )
            compared_differences += item["differences"]
    if append["total_differences"] != compared_differences:
        raise RuntimeError("TUSI-168 future append difference total drift")
    if checks["future_append_selection_differences_zero"] is not (
        append["total_differences"] == 0
    ):
        raise RuntimeError("TUSI-168 future append/check consistency drift")
    overlap = _require_exact_keys(
        report["control_overlap"], set(CONTROL_ORDER[1:]), "control overlap"
    )
    for control, value in overlap.items():
        item = _require_exact_keys(
            value,
            {
                "accepted",
                "exact_entry_intersection",
                "exact_entry_union",
                "exact_entry_jaccard",
                "exact_parent_identity_intersection",
            },
            f"{control} overlap",
        )
        jaccard = _require_exact_keys(
            item["exact_entry_jaccard"],
            {"numerator", "denominator"},
            f"{control} Jaccard",
        )
        if (
            any(
                isinstance(item[key], bool)
                or not isinstance(item[key], int)
                or item[key] < 0
                for key in (
                    "accepted",
                    "exact_entry_intersection",
                    "exact_entry_union",
                    "exact_parent_identity_intersection",
                )
            )
            or item["accepted"] != accepted_counts[control]
            or any(
                isinstance(jaccard[key], bool)
                or not isinstance(jaccard[key], int)
                or jaccard[key] < 0
                for key in ("numerator", "denominator")
            )
            or item["exact_entry_intersection"]
            > min(accepted_counts["primary"], item["accepted"])
            or item["exact_entry_union"]
            < max(accepted_counts["primary"], item["accepted"])
            or item["exact_entry_union"]
            > accepted_counts["primary"] + item["accepted"]
            or (
                jaccard["numerator"],
                jaccard["denominator"],
            )
            != (
                item["exact_entry_intersection"],
                item["exact_entry_union"]
                if item["exact_entry_union"]
                else 1,
            )
            or item["exact_parent_identity_intersection"]
            > min(accepted_counts["primary"], item["accepted"])
            or control in SAME_PARENT_CONTROLS
            and item["exact_parent_identity_intersection"]
            != accepted_counts["primary"]
        ):
            raise RuntimeError("TUSI-168 control overlap count drift")
    _require_exact_keys(
        report["clock_artifacts"],
        {"primary_sha256", "controls_sha256"},
        "clock artifacts",
    )
    evidence = _require_exact_keys(
        report["evidence_boundary"],
        set(EVIDENCE_BOUNDARY),
        "evidence boundary",
    )
    if any(
        type(value) is not type(EVIDENCE_BOUNDARY[key])
        for key, value in evidence.items()
    ):
        raise RuntimeError("TUSI-168 evidence boundary type drift")
    expected_evidence = {
        **EVIDENCE_BOUNDARY,
        "source_rows_opened": source_contract["rows"],
    }
    if (
        dict(evidence) != expected_evidence
        or report["source_support_precedes_novelty"] is not True
        or report[
            "novelty_comparator_market_or_outcome_artifacts_opened"
        ]
        is not False
    ):
        raise RuntimeError("TUSI-168 novelty/outcome evidence boundary drift")


def _validate_production_report_provenance(
    report: Mapping[str, Any],
    source_frame: pd.DataFrame,
) -> None:
    registration = validate_preregistration(
        DEFAULT_PREREGISTRATION,
        production=True,
    )
    production_frame, source_audit = load_source_artifacts(
        csv_path=DEFAULT_SOURCE_CSV,
        manifest_path=DEFAULT_SOURCE_MANIFEST,
        production=True,
    )
    supplied_records = _source_manifest_records(
        validate_source_frame(source_frame)
    )
    production_records = _source_manifest_records(
        validate_source_frame(production_frame)
    )
    expected_source_contract = {
        "columns": list(SOURCE_COLUMNS),
        "rows": int(source_audit["source_rows"]),
        **{
            key: source_audit[key]
            for key in SOURCE_CONTRACT_AUDIT_KEYS
        },
    }
    if (
        supplied_records != production_records
        or report.get("registration")
        != {
            "manifest_hash": registration["manifest_hash"],
            "mode": "artifact",
        }
        or report.get("source_contract") != expected_source_contract
    ):
        raise RuntimeError("TUSI-168 production report provenance drift")


def _validate_output_generation(
    report_bytes: bytes,
    primary_bytes: bytes,
    control_bytes: bytes,
    *,
    source_frame: pd.DataFrame,
    artifact_eligible_required: bool = False,
) -> None:
    if not isinstance(artifact_eligible_required, bool):
        raise RuntimeError("TUSI-168 artifact requirement type drift")
    try:
        report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("TUSI-168 report is not canonical JSON") from None
    if not isinstance(report, Mapping):
        raise TypeError("TUSI-168 report is not an object")
    report = cast(Mapping[str, Any], report)
    if report.get("artifact_eligible") is not artifact_eligible_required:
        raise RuntimeError("TUSI-168 artifact eligibility authorization drift")
    artifact_eligible_authorized = False
    if artifact_eligible_required:
        _validate_production_report_provenance(report, source_frame)
        artifact_eligible_authorized = True
    primary_rows, primary_counts = _validate_clock_csv(primary_bytes, primary=True)
    control_rows, control_counts = _validate_clock_csv(control_bytes, primary=False)
    _validate_report_schema(
        report,
        primary_counts=primary_counts,
        control_counts=control_counts,
        artifact_eligible_authorized=artifact_eligible_authorized,
    )
    validated_source = validate_source_frame(source_frame)
    controls, raw_counts = _build_controls_from_validated(validated_source)
    expected_control_parts = [
        controls[name] for name in CONTROL_ORDER[1:] if not controls[name].empty
    ]
    expected_control_frame = (
        pd.concat(expected_control_parts, ignore_index=True)
        if expected_control_parts
        else _empty_clock()
    )
    append_passed, append_report = future_append_selection_invariance(
        source_frame
    )
    if (
        report["source_contract"]["rows"] != len(validated_source)
        or report["raw_candidate_counts"] != raw_counts
        or primary_bytes != _csv_bytes(controls["primary"])
        or control_bytes != _csv_bytes(expected_control_frame)
        or report["future_append_selection_invariance"] != append_report
        or report["control_overlap"] != control_overlap(controls)
        or report["support_checks"][
            "future_append_selection_differences_zero"
        ]
        is not append_passed
    ):
        raise RuntimeError("TUSI-168 source-derived output generation drift")
    primary_frame = pd.DataFrame(
        primary_rows, columns=pd.Index(CLOCK_COLUMNS)
    )
    delayed_frame = pd.DataFrame(
        [row for row in control_rows if row["control"] == "one_bar_delayed_entry"],
        columns=pd.Index(CLOCK_COLUMNS),
    )
    for frame in (primary_frame, delayed_frame):
        for field in ("entry_time_utc", "exit_time_utc"):
            frame[field] = pd.to_datetime(frame[field], utc=True)
    period_ranges = {**SPLITS, **DIAGNOSTIC_PERIODS}
    observed_periods = {
        name: _period_diagnostic(
            project_clock_to_period(primary_frame, *period_ranges[name])
        )
        for name in PERIOD_ORDER
    }
    if report["period_diagnostics"] != observed_periods:
        raise RuntimeError("TUSI-168 period diagnostic projection drift")
    ordered_entries = _series(primary_frame, "entry_time_utc").sort_values()
    gaps = ordered_entries.diff().dropna()
    observed_gap = (
        int(gaps.max().total_seconds()) if not gaps.empty else None
    )
    if (
        report["support_audit"]["maximum_full_entry_gap_seconds"]
        != observed_gap
        or report["support_checks"][
            "maximum_full_entry_gap_at_most_240_days"
        ]
        is not bool(
            len(primary_frame) >= 2
            and observed_gap is not None
            and observed_gap <= 240 * 86400
        )
        or report["support_checks"][
            "identity_bucket_amount_side_clock_reproducible"
        ]
        is not True
    ):
        raise RuntimeError("TUSI-168 primary clock support audit drift")
    observed_crossers = delayed_control_boundary_crossers(primary_frame, delayed_frame)
    if (
        report["support_audit"]["one_bar_delayed_entry_boundary_crossers"]
        != observed_crossers
    ):
        raise RuntimeError("TUSI-168 delayed containment report drift")
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if report.get("manifest_hash") != canonical_hash(
        core
    ) or report_bytes != _json_bytes(report):
        raise RuntimeError("TUSI-168 report canonical hash drift")
    clocks = report.get("clock_artifacts")
    if clocks != {
        "primary_sha256": hashlib.sha256(primary_bytes).hexdigest(),
        "controls_sha256": hashlib.sha256(control_bytes).hexdigest(),
    }:
        raise RuntimeError("TUSI-168 report/clock generation drift")


def publish_outputs(
    *,
    primary_output: str | Path,
    controls_output: str | Path,
    report_output: str | Path,
    primary_bytes: bytes,
    control_bytes: bytes,
    report_bytes: bytes,
    source_frame: pd.DataFrame,
) -> dict[str, str]:
    outputs = {
        "primary": (_normalize_output_target(primary_output), primary_bytes),
        "controls": (_normalize_output_target(controls_output), control_bytes),
        "report": (_normalize_output_target(report_output), report_bytes),
    }
    for target, _ in outputs.values():
        _check_output_target(target)
    if len({target.path for target, _ in outputs.values()}) != len(outputs):
        raise RuntimeError("TUSI-168 output paths must be distinct")
    canonical_paths = {
        "primary": _normalize_output_target(DEFAULT_PRIMARY_OUTPUT).path,
        "controls": _normalize_output_target(DEFAULT_CONTROLS_OUTPUT).path,
        "report": _normalize_output_target(DEFAULT_REPORT_OUTPUT).path,
    }
    canonical_matches = {
        name: outputs[name][0].path == canonical_paths[name] for name in outputs
    }
    canonical_generation = all(canonical_matches.values())
    requested_paths = {target.path for target, _ in outputs.values()}
    if requested_paths & set(canonical_paths.values()) and not canonical_generation:
        raise RuntimeError("TUSI-168 mixed canonical output paths")
    _validate_output_generation(
        report_bytes,
        primary_bytes,
        control_bytes,
        source_frame=source_frame,
        artifact_eligible_required=canonical_generation,
    )
    existing = {
        name: _read_existing_output(target)
        for name, (target, _) in outputs.items()
    }
    for name, (_, expected) in outputs.items():
        if existing[name] is not None and existing[name] != expected:
            raise RuntimeError("TUSI-168 existing output generation drift")
    if all(value is not None for value in existing.values()):
        return {name: "verified_existing" for name in outputs}

    parent_fds: dict[Path, int] = {}
    staged: dict[str, str] = {}
    created: list[tuple[_OutputTarget, str]] = []
    statuses: dict[str, str] = {}
    try:
        for target, _ in outputs.values():
            if target.parent not in parent_fds:
                parent_fd = _open_output_parent(target.parent, create=True)
                if parent_fd is None:
                    raise RuntimeError("TUSI-168 output parent creation failed")
                parent_fds[target.parent] = parent_fd
        for name, (target, expected) in outputs.items():
            parent_fd = parent_fds[target.parent]
            observed = _read_output_at(parent_fd, target.leaf)
            if observed is not None and observed != expected:
                raise RuntimeError("TUSI-168 publication race drift")
            if observed is not None:
                statuses[name] = "verified_existing"
            else:
                staged[name] = _stage(parent_fd, target.leaf, expected)
                statuses[name] = "created"
        for name in ("primary", "controls", "report"):
            if name not in staged:
                continue
            target, expected = outputs[name]
            parent_fd = parent_fds[target.parent]
            try:
                os.link(
                    staged[name],
                    target.leaf,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                if _read_output_at(parent_fd, target.leaf) != expected:
                    raise RuntimeError("TUSI-168 publication race drift") from None
                statuses[name] = "verified_existing"
            except BaseException:
                if _same_inode(parent_fd, staged[name], target.leaf):
                    created.append((target, staged[name]))
                raise
            else:
                created.append((target, staged[name]))
                _fsync_directory(target.parent, directory_fd=parent_fd)
    except BaseException as publication_error:
        rollback_failed = False
        for target, temporary in reversed(created):
            parent_fd = parent_fds[target.parent]
            try:
                if _same_inode(parent_fd, temporary, target.leaf):
                    os.unlink(target.leaf, dir_fd=parent_fd)
                    _fsync_directory(
                        target.parent,
                        directory_fd=parent_fd,
                    )
            except OSError:
                rollback_failed = True
        if rollback_failed:
            raise RuntimeError(
                "TUSI-168 crash-durable publication rollback failed"
            ) from publication_error
        raise
    finally:
        try:
            for name, temporary in staged.items():
                target, _ = outputs[name]
                try:
                    os.unlink(
                        temporary,
                        dir_fd=parent_fds[target.parent],
                    )
                except FileNotFoundError:
                    pass
        finally:
            for descriptor in parent_fds.values():
                os.close(descriptor)
    return statuses


def evaluate_and_write(
    *,
    source_csv: str | Path,
    source_manifest: str | Path,
    preregistration: str | Path,
    primary_output: str | Path,
    controls_output: str | Path,
    report_output: str | Path,
    production: bool,
    registration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if production:
        frozen = (
            (Path(source_csv), DEFAULT_SOURCE_CSV),
            (Path(source_manifest), DEFAULT_SOURCE_MANIFEST),
            (Path(preregistration), DEFAULT_PREREGISTRATION),
            (Path(primary_output), DEFAULT_PRIMARY_OUTPUT),
            (Path(controls_output), DEFAULT_CONTROLS_OUTPUT),
            (Path(report_output), DEFAULT_REPORT_OUTPUT),
        )
        if any(actual != expected for actual, expected in frozen):
            raise RuntimeError("TUSI-168 production paths are frozen")
        if registration is not None:
            raise RuntimeError(
                "TUSI-168 production registration injection is forbidden"
            )
        _assert_evaluator_committed()
    registration = validate_preregistration(
        preregistration,
        registration=registration,
        production=production,
    )
    frame, source_audit = load_source_artifacts(
        csv_path=source_csv,
        manifest_path=source_manifest,
        production=production,
    )
    report, primary_bytes, control_bytes = _build_support_from_frame(
        frame,
        registration=registration,
        source_audit=source_audit,
        artifact_eligible=production,
    )
    report_bytes = _json_bytes(report)
    statuses = publish_outputs(
        primary_output=primary_output,
        controls_output=controls_output,
        report_output=report_output,
        primary_bytes=primary_bytes,
        control_bytes=control_bytes,
        report_bytes=report_bytes,
        source_frame=frame,
    )
    return {
        **statuses,
        "support_passed": report["support_passed"],
        "decision": report["decision"],
        "manifest_hash": report["manifest_hash"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--primary-output", required=True)
    parser.add_argument("--controls-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args(argv)
    if args.production:
        frozen = (
            (Path(args.source_csv), DEFAULT_SOURCE_CSV),
            (Path(args.source_manifest), DEFAULT_SOURCE_MANIFEST),
            (Path(args.preregistration), DEFAULT_PREREGISTRATION),
            (Path(args.primary_output), DEFAULT_PRIMARY_OUTPUT),
            (Path(args.controls_output), DEFAULT_CONTROLS_OUTPUT),
            (Path(args.report_output), DEFAULT_REPORT_OUTPUT),
        )
        if any(actual != expected for actual, expected in frozen):
            raise RuntimeError("TUSI-168 production CLI paths are frozen")
    result = evaluate_and_write(
        source_csv=args.source_csv,
        source_manifest=args.source_manifest,
        preregistration=args.preregistration,
        primary_output=args.primary_output,
        controls_output=args.controls_output,
        report_output=args.report_output,
        production=args.production,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
