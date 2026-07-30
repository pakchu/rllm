"""Outcome-blind incidence and source-support evaluator for ESDI-288.

The module deliberately has no market, comparator, funding, or outcome loader.
Its computational interface accepts normalized epoch rows, which keeps the
frozen incidence logic testable before the real Ethereum source replay exists.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from training import build_ethereum_settlement_demand_impulse_source as source_builder
from training import preregister_ethereum_settlement_demand_impulse as prereg


POLICY_ID = "ESDI-288"
PROTOCOL_VERSION = "ethereum_settlement_demand_impulse_source_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/evaluate_ethereum_settlement_demand_impulse_source_support.py"
)
TEST_PATH = Path(
    "tests/test_evaluate_ethereum_settlement_demand_impulse_source_support.py"
)
SOURCE_BUILDER_PATH = Path(
    "training/build_ethereum_settlement_demand_impulse_source.py"
)
PREREGISTRATION_PATH = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba"
)
PREREGISTRATION_MANIFEST_HASH = (
    "d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a"
)

DEFAULT_SOURCE_MANIFEST = source_builder.DEFAULT_MANIFEST_OUTPUT
DEFAULT_RAW_SOURCE = source_builder.DEFAULT_RAW_OUTPUT
DEFAULT_EPOCH_SOURCE = source_builder.DEFAULT_EPOCH_OUTPUT
DEFAULT_REPLAY_CLAIM = source_builder.REPLAY_CLAIM_PATH
DEFAULT_REPORT_OUTPUT = Path(
    "results/ethereum_settlement_demand_impulse_source_support_2026-07-30.json"
)
DEFAULT_PRIMARY_CLOCK_OUTPUT = Path(
    "results/ethereum_settlement_demand_impulse_primary_clock_2026-07-30.csv.gz"
)
DEFAULT_CONTROL_CLOCK_OUTPUT = Path(
    "results/ethereum_settlement_demand_impulse_control_clocks_2026-07-30.csv.gz"
)
DEFAULT_ATTEMPT_CLAIM = Path(
    "results/"
    "ethereum_settlement_demand_impulse_source_support_attempt_claim_2026-07-30.json"
)
ATTEMPT_CLAIM_PROTOCOL = (
    "ethereum_settlement_demand_impulse_source_support_attempt_claim_v1"
)

BAR = pd.Timedelta(minutes=5)
HOLD = pd.Timedelta(hours=24)
FULL_START = pd.Timestamp("2023-06-01T00:00:00Z")
FULL_END = pd.Timestamp("2026-06-01T00:00:00Z")
SPLITS = {
    "selection": (FULL_START, pd.Timestamp("2025-01-01T00:00:00Z")),
    "future25": (
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:00:00Z"),
    ),
    "future26": (pd.Timestamp("2026-01-01T00:00:00Z"), FULL_END),
}
SELECTION_REPORTS = {
    "2023H2": (FULL_START, pd.Timestamp("2024-01-01T00:00:00Z")),
    "2024H1": (
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-07-01T00:00:00Z"),
    ),
    "2024H2": (
        pd.Timestamp("2024-07-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
    ),
}

SOURCE_COLUMNS = tuple(
    (
        "epoch_id",
        "start_block",
        "end_block",
        "end_block_hash",
        "end_block_timestamp_utc",
        "confirmation_block",
        "confirmation_block_hash",
        "available_at_utc",
        "median_base_fee_wei_x2",
        "base_fee_vector_sha256",
        "mean_gas_used_ratio_decimal",
    )
)
FEATURE_COLUMNS = (
    "epoch_id",
    "available_at_utc",
    "source_hash",
    "primary_sign",
    "primary_ratio_num",
    "primary_ratio_den",
    "primary_rank_L",
    "primary_rank_E",
    "primary_rank_n",
    "primary_rank",
    "stale_sign",
    "stale_ratio_num",
    "stale_ratio_den",
    "stale_rank_L",
    "stale_rank_E",
    "stale_rank_n",
    "stale_rank",
    "gas_sign",
    "gas_ratio_num",
    "gas_ratio_den",
    "gas_rank_L",
    "gas_rank_E",
    "gas_rank_n",
    "gas_rank",
)
CLOCK_COLUMNS = (
    "policy_id",
    "control",
    "window",
    "signal_id",
    "epoch_id",
    "source_hash",
    "source_available_at_utc",
    "entry_time_utc",
    "exit_time_utc",
    "side",
    "rank_L",
    "rank_E",
    "rank_n",
    "rank_numerator",
    "rank_denominator",
)
INDEPENDENT_CONTROLS = (
    "primary",
    "base_fee_one_epoch_stale",
    "gas_utilization_only",
    "base_fee_no_tail",
)
SAME_PARENT_CONTROLS = (
    "exact_direction_flip",
    "deterministic_random_side",
    "constant_long",
    "constant_short",
    "one_bar_delayed_entry",
)
CONTROL_ORDER = INDEPENDENT_CONTROLS + SAME_PARENT_CONTROLS
FORBIDDEN_TOKENS = (
    "open",
    "high",
    "low",
    "close",
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
)
EVIDENCE_BOUNDARY = {
    "official_ethereum_raw_rows_opened": 0,
    "official_ethereum_epoch_rows_opened": 0,
    "synthetic_epoch_rows_processed": 0,
    "comparator_rows_opened": 0,
    "market_rows_opened": 0,
    "funding_rows_opened": 0,
    "outcome_rows_opened": 0,
    "outcomes_computed": False,
    "network_calls": 0,
}


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


def validate_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise RuntimeError("ESDI-288 preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION_PATH).read_text(encoding="utf-8"))
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("ESDI-288 preregistration manifest hash drift")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("ESDI-288 preregistration canonical hash drift")
    prereg.validate_manifest(payload)
    if any(payload.get(name) is not False for name in prereg.EVIDENCE_BOUNDARIES):
        raise RuntimeError("ESDI-288 preregistration evidence boundary is open")
    return payload


def _assert_protocol_committed() -> None:
    paths = [str(SCRIPT_PATH), str(TEST_PATH)]
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *paths],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if tracked.returncode:
        raise RuntimeError("ESDI-288 source-support evaluator is not committed")
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *paths],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if clean.returncode:
        raise RuntimeError("ESDI-288 source-support evaluator differs from HEAD")


def _format_time(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("ESDI-288 timestamps must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.microsecond or timestamp.nanosecond:
        raise RuntimeError("ESDI-288 timestamps must be whole-second")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(series: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="raise")
    if parsed.isna().any():
        raise RuntimeError(f"ESDI-288 null timestamp: {name}")
    if any(value.microsecond or value.nanosecond for value in parsed):
        raise RuntimeError(f"ESDI-288 subsecond timestamp: {name}")
    return parsed


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise RuntimeError("ESDI-288 gas ratio must be exact decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise RuntimeError("ESDI-288 gas ratio must be exact decimal") from None
    if not result.is_finite() or result < 0 or result > 1:
        raise RuntimeError("ESDI-288 gas ratio outside [0,1]")
    return result


def _reject_forbidden_columns(columns: Iterable[str]) -> None:
    for column in columns:
        lowered = str(column).lower()
        if any(token in lowered for token in FORBIDDEN_TOKENS):
            raise RuntimeError(f"ESDI-288 outcome-like column rejected: {column}")


def validate_epoch_frame(
    frame: pd.DataFrame,
    *,
    exact_domain: bool = False,
) -> pd.DataFrame:
    """Validate only the frozen normalized source contract."""

    if list(frame.columns) != list(SOURCE_COLUMNS):
        raise RuntimeError("ESDI-288 source exact schema drift")
    _reject_forbidden_columns(frame.columns)
    rows = frame.copy()
    if rows.empty:
        raise RuntimeError("ESDI-288 source is empty")
    for column in ("end_block_timestamp_utc", "available_at_utc"):
        rows[column] = _parse_time(rows[column], column)
    integer_columns = (
        "epoch_id",
        "start_block",
        "end_block",
        "confirmation_block",
        "median_base_fee_wei_x2",
    )
    for column in integer_columns:
        if rows[column].map(lambda value: isinstance(value, bool)).any():
            raise RuntimeError(f"ESDI-288 noninteger source field: {column}")
        numeric = pd.to_numeric(rows[column], errors="raise")
        if not numeric.map(lambda value: float(value).is_integer()).all():
            raise RuntimeError(f"ESDI-288 noninteger source field: {column}")
        rows[column] = numeric.astype("int64")
    if rows["epoch_id"].duplicated().any() or not rows[
        "epoch_id"
    ].is_monotonic_increasing:
        raise RuntimeError("ESDI-288 epoch identity/order drift")
    if not rows["available_at_utc"].is_monotonic_increasing:
        raise RuntimeError("ESDI-288 availability order drift")
    for row in rows.itertuples(index=False):
        expected_start, expected_end, expected_confirmation = prereg.epoch_blocks(
            int(row.epoch_id)
        )
        if (
            row.start_block != expected_start
            or row.end_block != expected_end
            or row.confirmation_block != expected_confirmation
        ):
            raise RuntimeError("ESDI-288 epoch block boundary drift")
        if row.median_base_fee_wei_x2 <= 0:
            raise RuntimeError("ESDI-288 median2 must be positive")
        if row.available_at_utc < row.end_block_timestamp_utc:
            raise RuntimeError("ESDI-288 availability precedes epoch end")
        for value, name in (
            (row.end_block_hash, "end_block_hash"),
            (row.confirmation_block_hash, "confirmation_block_hash"),
            (row.base_fee_vector_sha256, "base_fee_vector_sha256"),
        ):
            text = str(value)
            expected = 66 if name != "base_fee_vector_sha256" else 64
            if len(text) != expected or (
                name != "base_fee_vector_sha256" and not text.startswith("0x")
            ):
                raise RuntimeError(f"ESDI-288 malformed {name}")
            try:
                int(text[2:] if text.startswith("0x") else text, 16)
            except ValueError:
                raise RuntimeError(f"ESDI-288 malformed {name}") from None
    rows["mean_gas_used_ratio_decimal"] = rows[
        "mean_gas_used_ratio_decimal"
    ].map(_decimal)
    deltas = rows["epoch_id"].diff().dropna()
    if exact_domain and (
        len(rows) != prereg.LAST_EPOCH_ID - prereg.FIRST_EPOCH_ID + 1
        or rows["epoch_id"].iloc[0] != prereg.FIRST_EPOCH_ID
        or rows["epoch_id"].iloc[-1] != prereg.LAST_EPOCH_ID
        or not deltas.eq(1).all()
    ):
        raise RuntimeError("ESDI-288 exact source domain drift")
    return rows


def _magnitude_ratio(left: Any, right: Any) -> tuple[int, int] | None:
    if isinstance(left, Decimal) or isinstance(right, Decimal):
        a = Fraction(left)
        b = Fraction(right)
        if min(a, b) <= 0:
            return None
        value = max(a, b) / min(a, b)
        return value.numerator, value.denominator
    a, b = int(left), int(right)
    if min(a, b) <= 0:
        return None
    divisor = math.gcd(max(a, b), min(a, b))
    return max(a, b) // divisor, min(a, b) // divisor


def _sign(left: Any, right: Any) -> int:
    return (left > right) - (left < right)


def _rank_parts(
    current: tuple[int, int] | None,
    history: Sequence[tuple[int, int]],
) -> tuple[int, int, int, Fraction | None]:
    if current is None or len(history) < prereg.RANK_LOOKBACK:
        return -1, -1, len(history), None
    prior = list(history[-prereg.RANK_LOOKBACK :])
    rank = prereg.exact_rational_midrank(current, prior)
    lower = sum(prereg.compare_rationals(value, current) < 0 for value in prior)
    equal = sum(prereg.compare_rationals(value, current) == 0 for value in prior)
    return lower, equal, prereg.RANK_LOOKBACK, rank


def build_features(
    frame: pd.DataFrame,
    *,
    exact_domain: bool = False,
) -> pd.DataFrame:
    rows = validate_epoch_frame(frame, exact_domain=exact_domain)
    by_epoch = {int(row.epoch_id): row for row in rows.itertuples(index=False)}
    histories: dict[str, list[tuple[int, int]]] = {
        "primary": [],
        "stale": [],
        "gas": [],
    }
    output: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        epoch = int(row.epoch_id)
        lag2 = by_epoch.get(epoch - 2)
        lag1 = by_epoch.get(epoch - 1)
        lag3 = by_epoch.get(epoch - 3)
        primary_ratio = (
            _magnitude_ratio(row.median_base_fee_wei_x2, lag2.median_base_fee_wei_x2)
            if lag2 is not None
            else None
        )
        stale_ratio = (
            _magnitude_ratio(lag1.median_base_fee_wei_x2, lag3.median_base_fee_wei_x2)
            if lag1 is not None and lag3 is not None
            else None
        )
        gas_ratio = (
            _magnitude_ratio(
                row.mean_gas_used_ratio_decimal,
                lag2.mean_gas_used_ratio_decimal,
            )
            if lag2 is not None
            else None
        )
        ratios = {
            "primary": primary_ratio,
            "stale": stale_ratio,
            "gas": gas_ratio,
        }
        ranks = {
            name: _rank_parts(ratio, histories[name])
            for name, ratio in ratios.items()
        }
        for name, ratio in ratios.items():
            if ratio is not None:
                histories[name].append(ratio)

        def parts(name: str) -> tuple[int, int, int, Fraction | None]:
            return ranks[name]

        p_l, p_e, p_n, p_rank = parts("primary")
        s_l, s_e, s_n, s_rank = parts("stale")
        g_l, g_e, g_n, g_rank = parts("gas")
        output.append(
            {
                "epoch_id": epoch,
                "available_at_utc": row.available_at_utc,
                "source_hash": row.base_fee_vector_sha256,
                "primary_sign": (
                    _sign(row.median_base_fee_wei_x2, lag2.median_base_fee_wei_x2)
                    if lag2 is not None
                    else 0
                ),
                "primary_ratio_num": primary_ratio[0] if primary_ratio else None,
                "primary_ratio_den": primary_ratio[1] if primary_ratio else None,
                "primary_rank_L": p_l,
                "primary_rank_E": p_e,
                "primary_rank_n": p_n,
                "primary_rank": p_rank,
                "stale_sign": (
                    _sign(lag1.median_base_fee_wei_x2, lag3.median_base_fee_wei_x2)
                    if lag1 is not None and lag3 is not None
                    else 0
                ),
                "stale_ratio_num": stale_ratio[0] if stale_ratio else None,
                "stale_ratio_den": stale_ratio[1] if stale_ratio else None,
                "stale_rank_L": s_l,
                "stale_rank_E": s_e,
                "stale_rank_n": s_n,
                "stale_rank": s_rank,
                "gas_sign": (
                    _sign(
                        row.mean_gas_used_ratio_decimal,
                        lag2.mean_gas_used_ratio_decimal,
                    )
                    if lag2 is not None
                    else 0
                ),
                "gas_ratio_num": gas_ratio[0] if gas_ratio else None,
                "gas_ratio_den": gas_ratio[1] if gas_ratio else None,
                "gas_rank_L": g_l,
                "gas_rank_E": g_e,
                "gas_rank_n": g_n,
                "gas_rank": g_rank,
            }
        )
    return pd.DataFrame(output, columns=FEATURE_COLUMNS)


def _entry_time(availability: Any) -> pd.Timestamp:
    seconds = int(pd.Timestamp(availability).timestamp())
    return pd.Timestamp(prereg.ceil_5m_plus_one_bar(seconds), unit="s", tz="UTC")


def _side(sign: int) -> str:
    if sign == 1:
        return "LONG"
    if sign == -1:
        return "SHORT"
    raise RuntimeError("ESDI-288 zero has no side")


def _candidate(
    row: Any,
    control: str,
    sign: int,
    rank: Fraction | None,
    lower: int,
    equal: int,
    count: int,
) -> dict[str, Any]:
    entry = _entry_time(row.available_at_utc)
    return {
        "policy_id": POLICY_ID,
        "control": control,
        "window": None,
        "signal_id": prereg.canonical_signal_id(int(row.epoch_id)),
        "epoch_id": int(row.epoch_id),
        "source_hash": row.source_hash,
        "source_available_at_utc": row.available_at_utc,
        "entry_time_utc": entry,
        "exit_time_utc": entry + HOLD,
        "side": _side(sign),
        "rank_L": lower if rank is not None else None,
        "rank_E": equal if rank is not None else None,
        "rank_n": count if rank is not None else None,
        "rank_numerator": 2 * lower + equal if rank is not None else None,
        "rank_denominator": 2 * count if rank is not None else None,
    }


def raw_candidates(features: pd.DataFrame, control: str) -> pd.DataFrame:
    if control not in INDEPENDENT_CONTROLS:
        raise ValueError("ESDI-288 raw candidates require independent control")
    prefix = {
        "primary": "primary",
        "base_fee_one_epoch_stale": "stale",
        "gas_utilization_only": "gas",
        "base_fee_no_tail": "primary",
    }[control]
    result: list[dict[str, Any]] = []
    for row in features.itertuples(index=False):
        sign = int(getattr(row, f"{prefix}_sign"))
        rank = getattr(row, f"{prefix}_rank")
        if control == "base_fee_no_tail":
            accepted = sign != 0
        else:
            accepted = (
                sign != 0
                and getattr(row, f"{prefix}_rank_n") == prereg.RANK_LOOKBACK
                and isinstance(rank, Fraction)
                and rank >= Fraction(3, 4)
            )
        if accepted:
            result.append(
                _candidate(
                    row,
                    control,
                    sign,
                    None if control == "base_fee_no_tail" else rank,
                    int(getattr(row, f"{prefix}_rank_L")),
                    int(getattr(row, f"{prefix}_rank_E")),
                    int(getattr(row, f"{prefix}_rank_n")),
                )
            )
    return pd.DataFrame(result, columns=CLOCK_COLUMNS)


def _assign_window(rows: pd.DataFrame) -> pd.DataFrame:
    assigned: list[pd.Series] = []
    for _, row in rows.iterrows():
        for name, (start, end) in SPLITS.items():
            if row["entry_time_utc"] >= start and row["exit_time_utc"] <= end:
                copied = row.copy()
                copied["window"] = name
                assigned.append(copied)
                break
    return (
        pd.DataFrame(assigned, columns=CLOCK_COLUMNS)
        if assigned
        else pd.DataFrame(columns=CLOCK_COLUMNS)
    )


def reserve_nonoverlap(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply full/split containment, then one chronological global clock."""

    contained = _assign_window(rows)
    if contained.empty:
        return contained
    ordered = contained.sort_values(
        ["entry_time_utc", "source_available_at_utc", "epoch_id", "side"],
        kind="mergesort",
    )
    accepted: list[pd.Series] = []
    prior_exit: pd.Timestamp | None = None
    for _, row in ordered.iterrows():
        if prior_exit is None or row["entry_time_utc"] >= prior_exit:
            accepted.append(row)
            prior_exit = pd.Timestamp(row["exit_time_utc"])
    return pd.DataFrame(accepted, columns=CLOCK_COLUMNS).reset_index(drop=True)


def _parent_control(primary: pd.DataFrame, control: str) -> pd.DataFrame:
    if control not in SAME_PARENT_CONTROLS:
        raise ValueError(f"ESDI-288 unknown parent control: {control}")
    rows = primary.copy()
    rows["control"] = control
    if control == "exact_direction_flip":
        rows["side"] = rows["side"].map({"LONG": "SHORT", "SHORT": "LONG"})
    elif control == "deterministic_random_side":
        rows["side"] = rows["epoch_id"].map(prereg.deterministic_random_side)
    elif control == "constant_long":
        rows["side"] = "LONG"
    elif control == "constant_short":
        rows["side"] = "SHORT"
    else:
        rows["entry_time_utc"] = rows["entry_time_utc"] + BAR
        rows["exit_time_utc"] = rows["exit_time_utc"] + BAR
    return rows.loc[:, CLOCK_COLUMNS].reset_index(drop=True)


def build_controls(
    features: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    controls: dict[str, pd.DataFrame] = {}
    raw_counts: dict[str, int] = {}
    for control in INDEPENDENT_CONTROLS:
        raw = raw_candidates(features, control)
        raw_counts[control] = len(raw)
        controls[control] = reserve_nonoverlap(raw)
    for control in SAME_PARENT_CONTROLS:
        raw_counts[control] = len(controls["primary"])
        controls[control] = _parent_control(controls["primary"], control)
    return controls, raw_counts


def exact_entry_jaccard(left: pd.DataFrame, right: pd.DataFrame) -> Fraction:
    a = set(int(value.timestamp()) for value in left["entry_time_utc"])
    b = set(int(value.timestamp()) for value in right["entry_time_utc"])
    union = a | b
    return Fraction(len(a & b), len(union)) if union else Fraction(0, 1)


def candidate_24h_containment(left: pd.DataFrame, right: pd.DataFrame) -> Fraction:
    a = [int(value.timestamp()) for value in left["entry_time_utc"]]
    b = [int(value.timestamp()) for value in right["entry_time_utc"]]
    if not a or not b:
        return Fraction(0, 1)
    return prereg.bidirectional_entry_containment(a, b, 86_400)


def independent_control_metrics(
    controls: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    primary = controls["primary"]
    metrics: dict[str, dict[str, Any]] = {}
    checks: dict[str, bool] = {}
    for control in INDEPENDENT_CONTROLS[1:]:
        jaccard = exact_entry_jaccard(primary, controls[control])
        containment = candidate_24h_containment(primary, controls[control])
        metrics[control] = {
            "exact_entry_jaccard": _fraction_json(jaccard),
            "candidate_24h_containment": _fraction_json(containment),
        }
        checks[f"{control}_exact_entry_jaccard_strict"] = jaccard < Fraction(9, 10)
        checks[f"{control}_candidate_24h_containment_strict"] = containment < Fraction(
            19, 20
        )
    return metrics, checks


def _fraction_json(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _clock_stats(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "total": 0,
            "LONG": 0,
            "SHORT": 0,
            "maximum_month_share": None,
            "monthly_counts": {},
        }
    months = rows["entry_time_utc"].dt.strftime("%Y-%m").value_counts().sort_index()
    return {
        "total": len(rows),
        "LONG": int(rows["side"].eq("LONG").sum()),
        "SHORT": int(rows["side"].eq("SHORT").sum()),
        "maximum_month_share": float(months.max() / len(rows)),
        "monthly_counts": {str(key): int(value) for key, value in months.items()},
    }


def _maximum_same_side_run(rows: pd.DataFrame) -> int:
    maximum = current = 0
    previous: str | None = None
    for side in rows.sort_values("entry_time_utc")["side"]:
        current = current + 1 if side == previous else 1
        maximum = max(maximum, current)
        previous = side
    return maximum


def _clock_identity_reproducible(rows: pd.DataFrame) -> bool:
    if rows.empty:
        return False
    unique_columns = (
        "signal_id",
        "epoch_id",
        "source_hash",
        "entry_time_utc",
        "exit_time_utc",
    )
    if any(rows[column].duplicated().any() for column in unique_columns):
        return False
    for row in rows.itertuples(index=False):
        if row.signal_id != prereg.canonical_signal_id(int(row.epoch_id)):
            return False
        if row.entry_time_utc != _entry_time(row.source_available_at_utc):
            return False
        if row.exit_time_utc != row.entry_time_utc + HOLD:
            return False
        if row.side not in {"LONG", "SHORT"}:
            return False
        if (
            row.rank_n != prereg.RANK_LOOKBACK
            or row.rank_numerator != 2 * row.rank_L + row.rank_E
            or row.rank_denominator != 2 * prereg.RANK_LOOKBACK
        ):
            return False
        source_hash = str(row.source_hash)
        if len(source_hash) != 64:
            return False
        try:
            int(source_hash, 16)
        except ValueError:
            return False
    return True


def support_checks(
    primary: pd.DataFrame,
    *,
    controls: Mapping[str, pd.DataFrame] | None = None,
    source_rows: int = 2_474,
    missing_epochs: int = 0,
    dual_replay_differences: int = 0,
    boundary_header_differences: int = 0,
    append_invariance_passed: bool = True,
    reproducible: bool = True,
) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any]]:
    stats = {
        name: _clock_stats(primary.loc[primary["window"].eq(name)])
        for name in SPLITS
    }
    selection = primary.loc[primary["window"].eq("selection")]
    period_counts = {
        name: int(
            (
                selection["entry_time_utc"].ge(start)
                & selection["entry_time_utc"].lt(end)
            ).sum()
        )
        for name, (start, end) in SELECTION_REPORTS.items()
    }
    ordered = primary.sort_values("entry_time_utc")
    gaps = ordered["entry_time_utc"].diff().dropna()
    max_gap = gaps.max() if not gaps.empty else None

    def month_share_at_most(name: str, numerator: int, denominator: int) -> bool:
        counts = stats[name]["monthly_counts"]
        total = stats[name]["total"]
        return bool(
            total
            and counts
            and max(counts.values()) * denominator <= total * numerator
        )

    checks = {
        "source_exact_epochs": source_rows == 2_474,
        "source_missing_epochs_zero": missing_epochs == 0,
        "source_dual_replay_differences_zero": dual_replay_differences == 0,
        "source_boundary_header_differences_zero": boundary_header_differences == 0,
        "future_append_selection_differences_zero": append_invariance_passed,
        "identity_clock_side_rank_tie_source_hash_reproducible": (
            reproducible and _clock_identity_reproducible(primary)
        ),
        "selection_total_min": stats["selection"]["total"] >= 45,
        "selection_2023H2_min": period_counts["2023H2"] >= 12,
        "selection_2024H1_min": period_counts["2024H1"] >= 12,
        "selection_2024H2_min": period_counts["2024H2"] >= 12,
        "selection_each_side_min": min(
            stats["selection"]["LONG"], stats["selection"]["SHORT"]
        )
        >= 14,
        "selection_maximum_month_share": month_share_at_most(
            "selection", 1, 5
        ),
        "future25_total_min": stats["future25"]["total"] >= 30,
        "future25_each_side_min": min(
            stats["future25"]["LONG"], stats["future25"]["SHORT"]
        )
        >= 8,
        "future25_maximum_month_share": month_share_at_most(
            "future25", 1, 4
        ),
        "future26_total_min": stats["future26"]["total"] >= 15,
        "future26_each_side_min": min(
            stats["future26"]["LONG"], stats["future26"]["SHORT"]
        )
        >= 4,
        "future26_maximum_month_share": month_share_at_most(
            "future26", 3, 10
        ),
        "maximum_accepted_entry_gap_days": bool(
            max_gap is not None and max_gap <= pd.Timedelta(days=90)
        ),
        "maximum_same_side_run": _maximum_same_side_run(primary) <= 12
        and not primary.empty,
    }
    metric_report: dict[str, Any] = {}
    if controls is None:
        checks["independent_control_metrics_present"] = False
    else:
        metric_controls = {**controls, "primary": primary}
        metric_report, metric_checks = independent_control_metrics(metric_controls)
        checks.update(metric_checks)
    audit = {
        "clock_stats": stats,
        "selection_report_counts": period_counts,
        "maximum_accepted_entry_gap_seconds": (
            int(max_gap.total_seconds()) if max_gap is not None else None
        ),
        "maximum_same_side_run": _maximum_same_side_run(primary),
        "independent_control_metrics": metric_report,
    }
    return audit, checks, metric_report


def _frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(_csv_bytes(frame, compress=False)).hexdigest()


def _feature_hash(frame: pd.DataFrame) -> str:
    rows: list[list[Any]] = []
    for row in frame.itertuples(index=False, name=None):
        encoded: list[Any] = []
        for value in row:
            if isinstance(value, pd.Timestamp):
                encoded.append(_format_time(value))
            elif isinstance(value, Fraction):
                encoded.append([value.numerator, value.denominator])
            elif value is None or pd.isna(value):
                encoded.append(None)
            else:
                encoded.append(value)
        rows.append(encoded)
    return canonical_hash({"columns": list(frame.columns), "rows": rows})


def future_append_selection_invariance(
    frame: pd.DataFrame,
) -> tuple[bool, dict[str, Any]]:
    validated = validate_epoch_frame(frame, exact_domain=False)
    full_features = build_features(validated)
    full_controls, _ = build_controls(full_features)
    expected = full_controls["primary"].loc[
        full_controls["primary"]["window"].eq("selection")
    ]
    cutoff = SPLITS["selection"][1]
    prefix = validated.loc[validated["available_at_utc"].lt(cutoff)].reset_index(
        drop=True
    )
    if prefix.empty:
        return False, {
            "passed": False,
            "reason": "no completed source prefix before selection end",
        }
    prefix_features = build_features(prefix)
    prefix_controls, _ = build_controls(prefix_features)
    rebuilt = prefix_controls["primary"].loc[
        prefix_controls["primary"]["window"].eq("selection")
    ]
    expected_hash = _frame_hash(expected)
    rebuilt_hash = _frame_hash(rebuilt)
    passed = expected_hash == rebuilt_hash
    return passed, {
        "passed": passed,
        "selection_end_utc": _format_time(cutoff),
        "full_rebuild_selection_rows": len(expected),
        "prefix_rebuild_selection_rows": len(rebuilt),
        "full_rebuild_selection_sha256": expected_hash,
        "prefix_rebuild_selection_sha256": rebuilt_hash,
    }


def _csv_bytes(frame: pd.DataFrame, *, compress: bool = True) -> bytes:
    serial = frame.loc[:, CLOCK_COLUMNS].copy()
    for column in (
        "source_available_at_utc",
        "entry_time_utc",
        "exit_time_utc",
    ):
        serial[column] = serial[column].map(_format_time)
    raw = serial.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if not compress:
        return raw
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as handle:
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


def build_support_from_frame(
    frame: pd.DataFrame,
    *,
    exact_domain: bool = False,
    source_audit: Mapping[str, Any] | None = None,
    attempt_claim: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bytes, bytes]:
    validated = validate_epoch_frame(frame, exact_domain=exact_domain)
    features = build_features(validated, exact_domain=False)
    reproduced_features = build_features(validated.copy(), exact_domain=False)
    feature_hash = _feature_hash(features)
    reproducible = feature_hash == _feature_hash(reproduced_features)
    controls, raw_counts = build_controls(features)
    append_passed, append_report = future_append_selection_invariance(validated)
    audit_values = dict(source_audit or {})
    artifact_eligible = bool(
        exact_domain and audit_values.get("artifact_eligible") is True
    )
    if artifact_eligible:
        for key in (
            "source_manifest_sha256",
            "source_manifest_hash",
            "raw_source_sha256",
            "epoch_csv_sha256",
            "pre_replay_protocol_seal",
            "replay_claim",
        ):
            if not audit_values.get(key):
                raise RuntimeError(
                    f"ESDI-288 artifact-eligible source audit lacks {key}"
                )
    support_audit, checks, _ = support_checks(
        controls["primary"],
        controls=controls,
        source_rows=len(validated),
        missing_epochs=int(audit_values.get("missing_epochs", 0)),
        dual_replay_differences=int(audit_values.get("dual_replay_differences", 0)),
        boundary_header_differences=int(
            audit_values.get("boundary_header_differences", 0)
        ),
        append_invariance_passed=append_passed,
        reproducible=reproducible,
    )
    passed = all(checks.values())
    primary_bytes = _csv_bytes(controls["primary"])
    all_controls = pd.DataFrame(
        [
            row
            for name in CONTROL_ORDER[1:]
            for row in controls[name].to_dict(orient="records")
        ],
        columns=CLOCK_COLUMNS,
    )
    control_bytes = _csv_bytes(all_controls)
    evidence_boundary = dict(EVIDENCE_BOUNDARY)
    if artifact_eligible:
        evidence_boundary["official_ethereum_raw_rows_opened"] = int(
            audit_values["raw_source_rows_decoded"]
        )
        evidence_boundary["official_ethereum_epoch_rows_opened"] = int(
            audit_values["epoch_csv_rows_decoded"]
        )
    else:
        evidence_boundary["synthetic_epoch_rows_processed"] = len(validated)
    if artifact_eligible:
        status = "support_passed_terminal" if passed else "retired_terminal"
        decision = (
            "SOURCE_SUPPORT_PASS"
            if passed
            else "RETIRE_ESDI_288_UNCHANGED_BEFORE_OUTCOMES"
        )
    else:
        status = "synthetic_only_nonpublishable"
        decision = "SYNTHETIC_ONLY_NO_SOURCE_SUPPORT_DECISION"
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "status": status,
        "terminal": artifact_eligible,
        "artifact_eligible": artifact_eligible,
        "decision": decision,
        "support_passed": passed,
        "preregistration": {
            "path": str(PREREGISTRATION_PATH),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "attempt_claim": (
            dict(attempt_claim)
            if attempt_claim is not None
            else {"mode": "synthetic_only"}
        ),
        "source_contract": {
            "columns": list(SOURCE_COLUMNS),
            "rows": len(validated),
            **audit_values,
        },
        "feature_rows": len(features),
        "feature_rank_tie_state_sha256": feature_hash,
        "raw_candidate_counts": raw_counts,
        "accepted_clock_counts": {
            name: len(controls[name]) for name in CONTROL_ORDER
        },
        "support_audit": support_audit,
        "support_checks": checks,
        "future_append_selection_invariance": append_report,
        "clock_artifacts": {
            "primary_sha256": hashlib.sha256(primary_bytes).hexdigest(),
            "controls_sha256": hashlib.sha256(control_bytes).hexdigest(),
        },
        "evidence_boundary": evidence_boundary,
        "later_stage_artifacts_opened": False,
    }
    return (
        {**core, "manifest_hash": canonical_hash(core)},
        primary_bytes,
        control_bytes,
    )


def _exact_mapping(
    actual: Any,
    expected: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise RuntimeError(f"ESDI-288 source manifest {label} drift")
    return actual


def _sha256_text(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64:
        raise RuntimeError(f"ESDI-288 {label} must be SHA256")
    try:
        int(text, 16)
    except ValueError:
        raise RuntimeError(f"ESDI-288 {label} must be SHA256") from None
    return text


def _expected_boundary_audit() -> list[dict[str, Any]]:
    return [
        {
            "utc": boundary["utc"],
            "first_block_at_or_after": boundary["first_block_at_or_after"],
            "previous_block": boundary["first_block_at_or_after"] - 1,
            "previous_timestamp_before_boundary": True,
            "current_timestamp_at_or_after_boundary": True,
            "parent_relation_exact": True,
            "hash_exact": True,
        }
        for boundary in source_builder.FROZEN_BOUNDARIES
    ]


def _validate_protocol_seal(
    payload: Mapping[str, Any],
    *,
    production: bool,
) -> None:
    if "pre_replay_protocol_seal" not in payload:
        raise RuntimeError("ESDI-288 pre-replay protocol seal is missing")
    if not all(
        hasattr(source_builder, name)
        for name in (
            "PROTOCOL_PATHS",
            "current_protocol_seal",
            "validate_protocol_seal",
        )
    ):
        raise RuntimeError("ESDI-288 source builder lacks protocol-seal API")
    recorded = payload["pre_replay_protocol_seal"]
    if production:
        result = source_builder.validate_protocol_seal(recorded)
        if result is False:
            raise RuntimeError(
                "ESDI-288 pre-replay protocol seal validation failed"
            )
        return
    if not isinstance(recorded, Mapping) or set(recorded) != {
        "protocol_version",
        "policy_id",
        "mode",
        "protocol_paths",
        "seal_hash",
    }:
        raise RuntimeError("ESDI-288 synthetic protocol seal schema drift")
    core = {key: value for key, value in recorded.items() if key != "seal_hash"}
    if (
        recorded["protocol_version"]
        != "ethereum_settlement_demand_impulse_synthetic_protocol_seal_v1"
        or recorded["policy_id"] != POLICY_ID
        or recorded["mode"] != "synthetic_only"
        or recorded["protocol_paths"]
        != [path.as_posix() for path in source_builder.PROTOCOL_PATHS]
        or recorded["seal_hash"] != canonical_hash(core)
    ):
        raise RuntimeError("ESDI-288 synthetic protocol seal drift")


def _validate_claim_binding(
    claim: Any,
    seal: Mapping[str, Any],
    *,
    production: bool,
    raw_path: Path,
    epoch_path: Path,
    manifest_path: Path,
) -> None:
    if not isinstance(claim, Mapping):
        raise RuntimeError("ESDI-288 source replay claim binding is not an object")
    if production:
        if set(claim) != {"path", "sha256", "claim_hash"} or claim.get(
            "path"
        ) != DEFAULT_REPLAY_CLAIM.as_posix():
            raise RuntimeError("ESDI-288 production replay claim binding drift")
        claim_sha256 = _sha256_text(claim["sha256"], "replay claim file hash")
        claim_hash = _sha256_text(claim["claim_hash"], "replay claim hash")
        claim_path = _path(DEFAULT_REPLAY_CLAIM)
        raw = claim_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != claim_sha256:
            raise RuntimeError("ESDI-288 replay claim file hash drift")
        payload = json.loads(raw)
        if payload != source_builder._production_claim_payload(seal):
            raise RuntimeError("ESDI-288 replay claim payload drift")
        core = {key: value for key, value in payload.items() if key != "claim_hash"}
        if canonical_hash(core) != claim_hash:
            raise RuntimeError("ESDI-288 replay claim canonical hash drift")
        canonical = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        if raw != canonical:
            raise RuntimeError("ESDI-288 replay claim serialization drift")
    elif set(claim) != {"path", "sha256", "synthetic_unpublished"} or (
        claim.get("path") is not None
        or claim.get("synthetic_unpublished") is not True
    ):
        raise RuntimeError("ESDI-288 synthetic replay claim binding drift")
    else:
        claim_hash = _sha256_text(
            claim["sha256"], "synthetic replay claim hash"
        )
        claim_core = {
            "mode": "synthetic_unpublished",
            "outputs": [str(raw_path), str(epoch_path), str(manifest_path)],
            "pre_replay_protocol_seal_hash": seal["seal_hash"],
        }
        if claim_hash != canonical_hash(claim_core):
            raise RuntimeError("ESDI-288 synthetic replay claim hash drift")


def _validate_source_manifest_contract(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    raw_path: Path,
    epoch_path: Path,
    production: bool,
    expected_raw_rows: int,
    expected_epoch_rows: int,
) -> None:
    required_keys = {
        "protocol_version",
        "policy_id",
        "status",
        "claim",
        "preregistration",
        "pre_replay_protocol_seal",
        "source_builder",
        "transports",
        "rpc",
        "range",
        "epochs",
        "validation",
        "outputs",
        "outcome_boundary",
        "manifest_hash",
    }
    if set(payload) != required_keys:
        raise RuntimeError("ESDI-288 source manifest exact top-level schema drift")
    if payload["protocol_version"] != source_builder.PROTOCOL_VERSION:
        raise RuntimeError("ESDI-288 source protocol version drift")
    if payload["policy_id"] != POLICY_ID:
        raise RuntimeError("ESDI-288 source manifest policy drift")
    if payload["status"] != "complete_outcome_blind_source_replay":
        raise RuntimeError("ESDI-288 source replay is not terminal-complete")
    _exact_mapping(
        payload["preregistration"],
        {
            "path": str(source_builder.PREREGISTRATION_PATH),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "preregistration",
    )
    _validate_protocol_seal(payload, production=production)
    _validate_claim_binding(
        payload["claim"],
        payload["pre_replay_protocol_seal"],
        production=production,
        raw_path=raw_path,
        epoch_path=epoch_path,
        manifest_path=manifest_path,
    )
    builder_hash = _sha256_text(
        payload["source_builder"].get("sha256")
        if isinstance(payload["source_builder"], Mapping)
        else None,
        "source builder hash",
    )
    _exact_mapping(
        payload["source_builder"],
        {"path": str(SOURCE_BUILDER_PATH), "sha256": builder_hash},
        "source_builder",
    )
    if builder_hash != sha256_file(SOURCE_BUILDER_PATH):
        raise RuntimeError("ESDI-288 source builder bytes differ from manifest")
    if payload["transports"] != list(source_builder.TRANSPORTS):
        raise RuntimeError("ESDI-288 source transports drift")
    _exact_mapping(
        payload["rpc"],
        {
            "methods": list(source_builder.RPC_METHODS),
            "attempts_per_request": 1,
            "retry": False,
            "backoff": False,
            "fallback": False,
            "resume": False,
            "request_chunk_blocks": source_builder.REQUEST_CHUNK_BLOCKS,
            "fee_history_requests_per_transport": source_builder.REQUEST_COUNT,
            "boundary_header_requests_per_transport": (
                source_builder.BOUNDARY_HEADER_REQUESTS
            ),
            "finalized_header_requests_per_transport": 1,
            "epoch_and_confirmation_header_requests_per_transport": (
                source_builder.EPOCH_HEADER_REQUESTS
            ),
            "total_requests_per_transport": (
                source_builder.TOTAL_RPC_REQUESTS_PER_TRANSPORT
            ),
        },
        "RPC contract",
    )
    _exact_mapping(
        payload["range"],
        {
            "first_requested_block": source_builder.FIRST_REQUESTED_BLOCK,
            "last_requested_block": source_builder.LAST_REQUESTED_BLOCK,
            "last_retained_block": source_builder.LAST_RETAINED_BLOCK,
            "terminal_padding_blocks_requested": (
                source_builder.TERMINAL_PADDING_BLOCKS
            ),
            "terminal_padding_first_block": (
                source_builder.LAST_RETAINED_BLOCK + 1
            ),
            "terminal_padding_last_block": source_builder.LAST_REQUESTED_BLOCK,
            "terminal_padding_disposition": (
                "discarded_before_epoch_normalization"
            ),
            "terminal_padding_entered_normalized_epochs": 0,
            "last_request_before_frozen_2026_06_boundary": True,
        },
        "range",
    )
    _exact_mapping(
        payload["epochs"],
        {
            "first_epoch_id": source_builder.FIRST_EPOCH_ID,
            "last_epoch_id": source_builder.LAST_EPOCH_ID,
            "epoch_size_blocks": source_builder.EPOCH_SIZE_BLOCKS,
            "rows": expected_epoch_rows,
            "confirmation_blocks_after_end": source_builder.CONFIRMATION_BLOCKS,
            "base_fee_vector_sha256_implementation": (
                "training.preregister_ethereum_settlement_demand_impulse."
                "base_fee_vector_sha256"
            ),
            "gas_ratio_arithmetic": "decimal",
            "gas_ratio_decimal_precision": (
                source_builder.GAS_RATIO_DECIMAL_PRECISION
            ),
        },
        "epochs",
    )
    validation = payload["validation"]
    if not isinstance(validation, Mapping) or set(validation) != {
        "chain_id",
        "boundary_audit",
        "common_finalized_head",
        "common_finalized_head_hash",
        "common_finalized_head_at_or_after_last_confirmation",
        "dual_provider_response_differences",
        "shortened_responses",
        "next_base_fee_overlap_differences",
        "epoch_end_header_differences",
        "confirmation_header_differences",
    }:
        raise RuntimeError("ESDI-288 source manifest validation schema drift")
    if (
        validation["chain_id"] != source_builder.CHAIN_ID
        or validation["boundary_audit"] != _expected_boundary_audit()
        or type(validation["common_finalized_head"]) is not int
        or validation["common_finalized_head"]
        < source_builder.LAST_CONFIRMATION_BLOCK
        or len(str(validation["common_finalized_head_hash"])) != 66
        or not str(validation["common_finalized_head_hash"]).startswith("0x")
        or validation["common_finalized_head_at_or_after_last_confirmation"]
        is not True
        or any(
            validation[key] != 0
            for key in (
                "dual_provider_response_differences",
                "shortened_responses",
                "next_base_fee_overlap_differences",
                "epoch_end_header_differences",
                "confirmation_header_differences",
            )
        )
    ):
        raise RuntimeError("ESDI-288 source manifest validation drift")
    try:
        int(str(validation["common_finalized_head_hash"])[2:], 16)
    except ValueError:
        raise RuntimeError(
            "ESDI-288 source manifest finalized hash drift"
        ) from None
    outputs = payload["outputs"]
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "raw_chunks",
        "normalized_epochs",
        "manifest",
    }:
        raise RuntimeError("ESDI-288 source manifest output schema drift")
    raw = outputs["raw_chunks"]
    normalized = outputs["normalized_epochs"]
    if not isinstance(raw, Mapping) or set(raw) != {
        "path",
        "format",
        "rows",
        "bytes",
        "sha256",
    }:
        raise RuntimeError("ESDI-288 raw source output schema drift")
    if not isinstance(normalized, Mapping) or set(normalized) != {
        "path",
        "format",
        "rows",
        "columns",
        "bytes",
        "sha256",
    }:
        raise RuntimeError("ESDI-288 normalized output schema drift")
    expected_raw_path = str(DEFAULT_RAW_SOURCE if production else raw_path)
    expected_epoch_path = str(DEFAULT_EPOCH_SOURCE if production else epoch_path)
    _exact_mapping(
        raw,
        {
            "path": expected_raw_path,
            "format": "deterministic gzip NDJSON",
            "rows": expected_raw_rows,
            "bytes": raw["bytes"],
            "sha256": _sha256_text(raw["sha256"], "raw source hash"),
        },
        "raw output",
    )
    _exact_mapping(
        normalized,
        {
            "path": expected_epoch_path,
            "format": "deterministic gzip CSV",
            "rows": expected_epoch_rows,
            "columns": list(SOURCE_COLUMNS),
            "bytes": normalized["bytes"],
            "sha256": _sha256_text(
                normalized["sha256"], "normalized source hash"
            ),
        },
        "normalized output",
    )
    if (
        type(raw["bytes"]) is not int
        or raw["bytes"] <= 0
        or type(normalized["bytes"]) is not int
        or normalized["bytes"] <= 0
    ):
        raise RuntimeError("ESDI-288 source output bytes must be positive integers")
    if not isinstance(outputs["manifest"], Mapping):
        raise RuntimeError("ESDI-288 source manifest output binding drift")
    _exact_mapping(
        outputs["manifest"],
        {
            "path": str(
                DEFAULT_SOURCE_MANIFEST
                if production
                else outputs["manifest"]["path"]
            )
        },
        "manifest output",
    )
    _exact_mapping(
        payload["outcome_boundary"],
        {
            "ethereum_source_values_opened": True,
            "btc_market_rows_opened": 0,
            "comparator_rows_opened": 0,
            "funding_rows_opened": 0,
            "return_or_pnl_rows_opened": 0,
            "outcomes_opened": False,
        },
        "outcome boundary",
    )


def _decoded_raw_rows(path: Path) -> int:
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n") or not line.strip():
                raise RuntimeError("ESDI-288 raw source row framing drift")
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise RuntimeError("ESDI-288 raw source row is not an object")
            count += 1
    return count


def _load_source_artifacts(
    *,
    manifest_path: Path,
    raw_path: Path,
    epoch_path: Path,
    production: bool,
    expected_raw_rows: int,
    expected_epoch_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_bytes = manifest_path.read_bytes()
    payload = json.loads(manifest_bytes)
    manifest_hash = _sha256_text(
        payload.get("manifest_hash")
        if isinstance(payload, Mapping)
        else None,
        "source manifest hash",
    )
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != manifest_hash:
        raise RuntimeError("ESDI-288 source manifest canonical hash drift")
    canonical_bytes = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if manifest_bytes != canonical_bytes:
        raise RuntimeError("ESDI-288 source manifest serialization drift")
    _validate_source_manifest_contract(
        payload,
        manifest_path=manifest_path,
        raw_path=raw_path,
        epoch_path=epoch_path,
        production=production,
        expected_raw_rows=expected_raw_rows,
        expected_epoch_rows=expected_epoch_rows,
    )
    raw_contract = payload["outputs"]["raw_chunks"]
    epoch_contract = payload["outputs"]["normalized_epochs"]
    raw_bytes = raw_path.read_bytes()
    epoch_bytes = epoch_path.read_bytes()
    if len(raw_bytes) != raw_contract["bytes"] or hashlib.sha256(
        raw_bytes
    ).hexdigest() != raw_contract["sha256"]:
        raise RuntimeError("ESDI-288 raw source bytes/hash drift")
    if len(epoch_bytes) != epoch_contract["bytes"] or hashlib.sha256(
        epoch_bytes
    ).hexdigest() != epoch_contract["sha256"]:
        raise RuntimeError("ESDI-288 normalized source bytes/hash drift")
    raw_rows = _decoded_raw_rows(raw_path)
    if raw_rows != raw_contract["rows"]:
        raise RuntimeError("ESDI-288 raw source decoded row count drift")
    with gzip.open(epoch_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError("ESDI-288 normalized source is empty") from None
        decoded_epoch_rows = sum(1 for _ in reader)
    if header != list(SOURCE_COLUMNS):
        _reject_forbidden_columns(header)
        raise RuntimeError("ESDI-288 normalized epoch CSV header drift")
    if decoded_epoch_rows != epoch_contract["rows"]:
        raise RuntimeError("ESDI-288 normalized decoded row count drift")
    frame = pd.read_csv(
        epoch_path,
        dtype={"mean_gas_used_ratio_decimal": "string"},
    )
    if list(frame.columns) != list(SOURCE_COLUMNS) or len(frame) != decoded_epoch_rows:
        raise RuntimeError("ESDI-288 normalized epoch CSV decode drift")
    validation = payload["validation"]
    audit = {
        "artifact_eligible": production,
        "source_manifest_path": str(
            DEFAULT_SOURCE_MANIFEST if production else manifest_path
        ),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_manifest_hash": manifest_hash,
        "raw_source_path": str(DEFAULT_RAW_SOURCE if production else raw_path),
        "raw_source_bytes": len(raw_bytes),
        "raw_source_rows_decoded": raw_rows,
        "raw_source_sha256": raw_contract["sha256"],
        "epoch_csv_path": str(
            DEFAULT_EPOCH_SOURCE if production else epoch_path
        ),
        "epoch_csv_bytes": len(epoch_bytes),
        "epoch_csv_rows_decoded": decoded_epoch_rows,
        "epoch_csv_sha256": epoch_contract["sha256"],
        "pre_replay_protocol_seal": payload["pre_replay_protocol_seal"],
        "replay_claim": payload["claim"],
        "missing_epochs": source_builder.EPOCH_COUNT - decoded_epoch_rows,
        "dual_replay_differences": validation[
            "dual_provider_response_differences"
        ],
        "boundary_header_differences": (
            validation["epoch_end_header_differences"]
            + validation["confirmation_header_differences"]
        ),
    }
    return frame, audit


def _git(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"ESDI-288 git verification failed: {' '.join(args)}"
        )
    return result.stdout


def _assert_production_source_artifacts_committed() -> None:
    paths = (
        DEFAULT_REPLAY_CLAIM,
        DEFAULT_SOURCE_MANIFEST,
        DEFAULT_RAW_SOURCE,
        DEFAULT_EPOCH_SOURCE,
    )
    for path in paths:
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(
                f"ESDI-288 canonical source artifact is unsafe: {path}"
            )
        resolved = REPOSITORY_ROOT
        for part in path.parts:
            resolved = resolved / part
            if resolved.is_symlink():
                raise RuntimeError(
                    f"ESDI-288 canonical source artifact is unsafe: {path}"
                )
        if not resolved.is_file():
            raise RuntimeError(
                f"ESDI-288 canonical source artifact is unsafe: {path}"
            )
    _git("ls-files", "--error-unmatch", "--", *(str(path) for path in paths))
    _git("diff", "--quiet", "HEAD", "--", *(str(path) for path in paths))
    commits = {
        _git("log", "-1", "--format=%H", "--", str(path))
        .decode("ascii")
        .strip()
        for path in paths
    }
    if len(commits) != 1:
        raise RuntimeError(
            "ESDI-288 canonical source artifacts lack one source-artifact commit"
        )
    artifact_commit = next(iter(commits))
    if len(artifact_commit) not in {40, 64}:
        raise RuntimeError("ESDI-288 source-artifact commit is missing")
    _git("merge-base", "--is-ancestor", artifact_commit, "HEAD")


def load_source_manifest() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only the committed, HEAD-clean canonical production artifacts."""

    _assert_production_source_artifacts_committed()
    return _load_source_artifacts(
        manifest_path=_path(DEFAULT_SOURCE_MANIFEST),
        raw_path=_path(DEFAULT_RAW_SOURCE),
        epoch_path=_path(DEFAULT_EPOCH_SOURCE),
        production=True,
        expected_raw_rows=source_builder.REQUEST_COUNT,
        expected_epoch_rows=source_builder.EPOCH_COUNT,
    )


def load_synthetic_source_artifacts(
    *,
    manifest_path: str | Path,
    raw_path: str | Path,
    epoch_path: str | Path,
    expected_raw_rows: int,
    expected_epoch_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Explicit non-production loader for synthetic contract tests only."""

    return _load_source_artifacts(
        manifest_path=_path(manifest_path),
        raw_path=_path(raw_path),
        epoch_path=_path(epoch_path),
        production=False,
        expected_raw_rows=expected_raw_rows,
        expected_epoch_rows=expected_epoch_rows,
    )


def write_once(path: str | Path, data: bytes) -> str:
    output = _path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != data:
            raise RuntimeError(f"ESDI-288 noncanonical existing artifact: {path}")
        return "verified_existing"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            if output.read_bytes() != data:
                raise RuntimeError(f"ESDI-288 artifact race drift: {path}")
            return "verified_existing"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def _validate_publication_generation(
    report_bytes: bytes,
    primary_bytes: bytes,
    control_bytes: bytes,
) -> None:
    try:
        report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("ESDI-288 report generation is not valid JSON") from None
    if not isinstance(report, Mapping):
        raise RuntimeError("ESDI-288 report generation is not an object")
    manifest_hash = _sha256_text(
        report.get("manifest_hash"), "support report manifest hash"
    )
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if canonical_hash(core) != manifest_hash or _json_bytes(report) != report_bytes:
        raise RuntimeError("ESDI-288 support report canonical hash drift")
    clocks = report.get("clock_artifacts")
    if not isinstance(clocks, Mapping) or set(clocks) != {
        "primary_sha256",
        "controls_sha256",
    }:
        raise RuntimeError("ESDI-288 support report clock hash schema drift")
    if clocks["primary_sha256"] != hashlib.sha256(primary_bytes).hexdigest() or clocks[
        "controls_sha256"
    ] != hashlib.sha256(control_bytes).hexdigest():
        raise RuntimeError("ESDI-288 support report describes a mixed generation")


def _stage_bytes(output: Path, data: bytes) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise RuntimeError("ESDI-288 output parent is not a real directory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())
    return temporary


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _attempt_claim_payload() -> dict[str, Any]:
    core = {
        "protocol_version": ATTEMPT_CLAIM_PROTOCOL,
        "policy_id": POLICY_ID,
        "status": "claimed_before_source_rows",
        "one_shot": True,
        "retry_or_repair_after_failure": False,
        "preregistration": {
            "path": str(PREREGISTRATION_PATH),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "canonical_inputs": [
            str(DEFAULT_REPLAY_CLAIM),
            str(DEFAULT_SOURCE_MANIFEST),
            str(DEFAULT_RAW_SOURCE),
            str(DEFAULT_EPOCH_SOURCE),
        ],
        "canonical_outputs": [
            str(DEFAULT_PRIMARY_CLOCK_OUTPUT),
            str(DEFAULT_CONTROL_CLOCK_OUTPUT),
            str(DEFAULT_REPORT_OUTPUT),
        ],
    }
    return {**core, "claim_hash": canonical_hash(core)}


def _claim_binding(raw: bytes, payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": str(DEFAULT_ATTEMPT_CLAIM),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "claim_hash": str(payload["claim_hash"]),
    }


def _create_attempt_claim() -> dict[str, str]:
    path = _path(DEFAULT_ATTEMPT_CLAIM)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError("ESDI-288 source-support claim parent is unsafe")
    if path.exists() or path.is_symlink():
        raise RuntimeError("ESDI-288 source-support attempt is already claimed")
    payload = _attempt_claim_payload()
    raw = _json_bytes(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("source-support claim write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return _claim_binding(raw, payload)


def load_attempt_claim() -> dict[str, str]:
    path = _path(DEFAULT_ATTEMPT_CLAIM)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("ESDI-288 source-support attempt claim is invalid")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "ESDI-288 source-support attempt claim is invalid"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or dict(payload) != _attempt_claim_payload()
        or raw != _json_bytes(payload)
    ):
        raise RuntimeError("ESDI-288 source-support attempt claim drift")
    return _claim_binding(raw, payload)


def publish_support_transaction(
    *,
    report_output: str | Path,
    primary_output: str | Path,
    controls_output: str | Path,
    report_bytes: bytes,
    primary_bytes: bytes,
    control_bytes: bytes,
) -> dict[str, str]:
    """Publish one generation, using the report as the completion marker."""

    _validate_publication_generation(report_bytes, primary_bytes, control_bytes)
    outputs = {
        "primary_clock_status": (_path(primary_output), primary_bytes),
        "control_clocks_status": (_path(controls_output), control_bytes),
        "report_status": (_path(report_output), report_bytes),
    }
    paths = [item[0] for item in outputs.values()]
    if len(set(paths)) != len(paths):
        raise RuntimeError("ESDI-288 support output paths must be distinct")
    report_path = outputs["report_status"][0]
    if report_path.exists() or report_path.is_symlink():
        if not all(path.exists() and not path.is_symlink() for path in paths):
            raise RuntimeError(
                "ESDI-288 report completion marker exists without both clocks"
            )
        for path, expected in (item for item in outputs.values()):
            if path.read_bytes() != expected:
                raise RuntimeError("ESDI-288 existing artifacts are mixed generation")
        return {name: "verified_existing" for name in outputs}

    statuses: dict[str, str] = {}
    staged: dict[str, Path] = {}
    created: list[Path] = []
    try:
        for name, (path, expected) in outputs.items():
            if path.is_symlink():
                raise RuntimeError(
                    "ESDI-288 support artifact may not be a symlink"
                )
            if path.exists():
                if path.read_bytes() != expected:
                    raise RuntimeError(
                        "ESDI-288 existing artifacts are mixed generation"
                    )
                statuses[name] = "verified_existing"
            else:
                staged[name] = _stage_bytes(path, expected)
                statuses[name] = "created"
        for name in (
            "primary_clock_status",
            "control_clocks_status",
            "report_status",
        ):
            if name not in staged:
                continue
            path = outputs[name][0]
            try:
                os.link(staged[name], path)
            except FileExistsError:
                if path.is_symlink() or path.read_bytes() != outputs[name][1]:
                    raise RuntimeError(
                        "ESDI-288 publication race produced mixed generation"
                    ) from None
                statuses[name] = "verified_existing"
            else:
                created.append(path)
                _fsync_directory(path.parent)
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
            _fsync_directory(path.parent)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return statuses


def _load_completed_support_generation(
    attempt_claim: Mapping[str, Any],
) -> dict[str, Any]:
    paths = (
        _path(DEFAULT_REPORT_OUTPUT),
        _path(DEFAULT_PRIMARY_CLOCK_OUTPUT),
        _path(DEFAULT_CONTROL_CLOCK_OUTPUT),
    )
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise RuntimeError(
            "ESDI-288 claimed source-support attempt lacks a complete generation"
        )
    report_bytes, primary_bytes, control_bytes = (
        path.read_bytes() for path in paths
    )
    _validate_publication_generation(
        report_bytes,
        primary_bytes,
        control_bytes,
    )
    report = json.loads(report_bytes)
    if (
        report.get("protocol_version") != PROTOCOL_VERSION
        or report.get("policy_id") != POLICY_ID
        or report.get("terminal") is not True
        or report.get("artifact_eligible") is not True
        or report.get("attempt_claim") != dict(attempt_claim)
    ):
        raise RuntimeError(
            "ESDI-288 claimed source-support completion binding drift"
        )
    return {
        "primary_clock_status": "verified_existing",
        "control_clocks_status": "verified_existing",
        "report_status": "verified_existing",
        "support_passed": bool(report["support_passed"]),
        "status": str(report["status"]),
        "decision": str(report["decision"]),
        "manifest_hash": str(report["manifest_hash"]),
    }


def write_support(
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
    primary_output: str | Path = DEFAULT_PRIMARY_CLOCK_OUTPUT,
    controls_output: str | Path = DEFAULT_CONTROL_CLOCK_OUTPUT,
) -> dict[str, Any]:
    if (
        Path(report_output) != DEFAULT_REPORT_OUTPUT
        or Path(primary_output) != DEFAULT_PRIMARY_CLOCK_OUTPUT
        or Path(controls_output) != DEFAULT_CONTROL_CLOCK_OUTPUT
    ):
        raise RuntimeError("ESDI-288 artifact paths are frozen")
    _assert_protocol_committed()
    validate_preregistration()
    output_paths = tuple(
        _path(path)
        for path in (
            DEFAULT_REPORT_OUTPUT,
            DEFAULT_PRIMARY_CLOCK_OUTPUT,
            DEFAULT_CONTROL_CLOCK_OUTPUT,
        )
    )
    claim_path = _path(DEFAULT_ATTEMPT_CLAIM)
    if claim_path.exists() or claim_path.is_symlink():
        return _load_completed_support_generation(load_attempt_claim())
    if any(path.exists() or path.is_symlink() for path in output_paths):
        raise RuntimeError(
            "ESDI-288 source-support output exists without its attempt claim"
        )
    attempt_claim = _create_attempt_claim()
    frame, source_audit = load_source_manifest()
    report, primary_bytes, control_bytes = build_support_from_frame(
        frame,
        exact_domain=True,
        source_audit=source_audit,
        attempt_claim=attempt_claim,
    )
    if report.get("artifact_eligible") is not True or report.get("terminal") is not True:
        raise RuntimeError("ESDI-288 production support report is not publishable")
    publication = publish_support_transaction(
        report_output=report_output,
        primary_output=primary_output,
        controls_output=controls_output,
        report_bytes=_json_bytes(report),
        primary_bytes=primary_bytes,
        control_bytes=control_bytes,
    )
    return {
        **publication,
        "support_passed": report["support_passed"],
        "status": report["status"],
        "decision": report["decision"],
        "manifest_hash": report["manifest_hash"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(write_support(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
