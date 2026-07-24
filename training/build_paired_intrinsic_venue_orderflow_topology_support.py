"""Build the outcome-blind PIVOT-72 source-support state clock."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import (
    preregister_paired_intrinsic_venue_orderflow_topology as prereg,
)


PROTOCOL_VERSION = "paired_intrinsic_venue_orderflow_topology_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/build_paired_intrinsic_venue_orderflow_topology_support.py"
)
TEST_PATH = Path(
    "tests/test_build_paired_intrinsic_venue_orderflow_topology_support.py"
)
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "e2a94c3e675addaad4bb0075a27f4155e5f1675bcc1552a66ceb9c30a5ceab28"
)
PREREGISTRATION_MANIFEST_HASH = (
    "09faeaffba6d7c88e420c2198dfb536f2b82c3198f210a0e166906bf5c1cb532"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "data/paired_intrinsic_venue_orderflow_topology_"
    "states_2020_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/paired_intrinsic_venue_orderflow_topology_"
    "support_2026-07-24.json"
)

BAR = pd.Timedelta(minutes=5)
DAY = pd.Timedelta(days=1)
SOURCE_START = pd.Timestamp("2020-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2024-01-01T00:00:00Z")
ROWS_PER_DAY = 288
LATEST_ANCHOR_INDEX = prereg.Policy().latest_anchor_start_minute_utc // 5

SPLITS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (
        SOURCE_START,
        pd.Timestamp("2022-01-01T00:00:00Z"),
    ),
    "selection": (
        pd.Timestamp("2022-01-01T00:00:00Z"),
        pd.Timestamp("2023-01-01T00:00:00Z"),
    ),
    "eval": (
        pd.Timestamp("2023-01-01T00:00:00Z"),
        SOURCE_END,
    ),
}

RAW_MEASURE_COLUMNS = (
    "gap_bars",
    "early_anchor_start_minute",
    "laggard_progress_at_early",
    "spot_flow_early",
    "um_flow_early",
    "spot_flow_late",
    "um_flow_late",
    "spot_abs_flow_late",
    "um_abs_flow_late",
)
QUARTILE_RAW_COLUMNS = {
    "gap_q": "gap_bars",
    "laggard_progress_q": "laggard_progress_at_early",
    "spot_late_abs_flow_q": "spot_abs_flow_late",
    "um_late_abs_flow_q": "um_abs_flow_late",
}
TIME_COLUMNS = (
    "spot_anchor_time",
    "um_anchor_time",
    "causal_origin_time",
    "state_completion_time",
    "buffer_completion_time",
    "decision_deadline_time",
    "entry_time",
    "exit_time",
)
BASE_COLUMNS = (
    "base_state_id",
    "source_day",
    "spot_anchor_index",
    "um_anchor_index",
    "early_index",
    "late_index",
    *TIME_COLUMNS,
    "leader_venue",
    *RAW_MEASURE_COLUMNS,
)
CLOCK_COLUMNS = (
    *BASE_COLUMNS,
    *prereg.TOKEN_COLUMNS,
    "primary_reserved",
    "primary_split",
)
FORBIDDEN_CLOCK_TOKENS = (
    "price",
    "return",
    "basis",
    "funding",
    "premium",
    "open_interest",
    "kimchi",
    "dxy",
    "reward",
    "action",
    "pnl",
    "cagr",
    "mdd",
    "comparator",
)


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


def _format_time(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("PIVOT timestamp must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.microsecond or timestamp.nanosecond:
        raise RuntimeError("PIVOT timestamp must be whole-second")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_day(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("PIVOT source day must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.floor("D"):
        raise RuntimeError("PIVOT source day must be UTC midnight")
    return timestamp.strftime("%Y-%m-%d")


def _git_check(*args: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("git")
    if executable is None:
        fallback = Path("/usr/bin/git")
        if not fallback.is_file():
            raise RuntimeError("PIVOT cannot resolve the git executable")
        executable = str(fallback)
    return subprocess.run(
        [executable, *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_protocol_committed() -> None:
    paths = (str(SCRIPT_PATH), str(TEST_PATH))
    tracked = _git_check("ls-files", "--error-unmatch", "--", *paths)
    if tracked.returncode:
        raise RuntimeError("PIVOT source-support protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *paths)
    if clean.returncode:
        raise RuntimeError("PIVOT source-support protocol differs from HEAD")


def validate_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("PIVOT preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_manifest(payload)
    if payload != prereg.build_manifest():
        raise RuntimeError("PIVOT preregistration differs from frozen builder")
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("PIVOT preregistration manifest hash drift")
    for field in (
        "outcomes_opened",
        "source_incidence_opened",
        "source_rows_decoded",
        "market_value_rows_decoded",
        "funding_value_rows_decoded",
        "comparator_rows_decoded",
        "post_2023_values_decoded",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"PIVOT preregistration boundary opened: {field}")
    if payload["source_contract"]["allowlist"] != list(
        prereg.SOURCE_ALLOWLIST
    ):
        raise RuntimeError("PIVOT source allowlist drift")
    return payload


def verify_pre_source_bindings() -> dict[str, dict[str, str]]:
    bindings = {
        str(PREREGISTRATION): PREREGISTRATION_SHA256,
        prereg.BOUNDARY_DOCUMENT: prereg.BOUNDARY_DOCUMENT_SHA256,
        prereg.MECHANISM_DOCUMENT: prereg.MECHANISM_DOCUMENT_SHA256,
        prereg.SOURCE: prereg.SOURCE_SHA256,
        prereg.SOURCE_MANIFEST: prereg.SOURCE_MANIFEST_SHA256,
        prereg.SOURCE_AUDIT: prereg.SOURCE_AUDIT_SHA256,
    }
    if any(path in bindings for path in prereg.FORBIDDEN_COMPARATOR_PATHS):
        raise RuntimeError("PIVOT forbidden comparator entered source bindings")
    audit: dict[str, dict[str, str]] = {}
    for path, expected in bindings.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"PIVOT frozen source binding changed: {path}")
        audit[path] = {"path": path, "sha256": actual}
    if prereg.sha256_csv_header(prereg.SOURCE) != prereg.SOURCE_HEADER_SHA256:
        raise RuntimeError("PIVOT source header hash drift")
    if not set(prereg.SOURCE_ALLOWLIST).issubset(
        prereg.csv_header(prereg.SOURCE)
    ):
        raise RuntimeError("PIVOT source allowlist differs from source header")
    return audit


def _strict_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    if not normalized.isin(("true", "false")).all():
        raise RuntimeError("PIVOT source_complete is not strict boolean")
    return normalized.eq("true")


def validate_source_frame(
    frame: pd.DataFrame,
    *,
    exact_grid: bool,
) -> pd.DataFrame:
    if list(frame.columns) != list(prereg.SOURCE_ALLOWLIST):
        raise RuntimeError("PIVOT source loader did not preserve exact allowlist")
    validated = frame.copy()
    for column in (
        "date",
        "feature_available_time_utc",
        "trade_earliest_time_utc",
    ):
        validated[column] = pd.to_datetime(
            validated[column],
            utc=True,
            errors="raise",
        )
    for column in (
        "spot_quote_notional",
        "um_quote_notional",
        "spot_signed_quote_notional",
        "um_signed_quote_notional",
    ):
        validated[column] = pd.to_numeric(
            validated[column],
            errors="coerce",
        ).astype(float)
    validated["source_complete"] = _strict_bool(
        validated["source_complete"]
    )
    if validated.empty:
        raise RuntimeError("PIVOT source is empty")
    if validated["date"].duplicated().any():
        raise RuntimeError("PIVOT source timestamps duplicated")
    if not validated["date"].is_monotonic_increasing:
        raise RuntimeError("PIVOT source timestamps not monotonic")
    if (
        validated["date"].iloc[0] < SOURCE_START
        or validated["date"].iloc[-1] >= SOURCE_END
    ):
        raise RuntimeError("PIVOT source timestamp outside frozen coverage")
    offsets = (validated["date"] - SOURCE_START) / BAR
    if not np.equal(offsets.to_numpy(dtype=float) % 1.0, 0.0).all():
        raise RuntimeError("PIVOT source timestamp is off the five-minute grid")
    expected_availability = validated["date"] + BAR
    if not validated["feature_available_time_utc"].equals(
        expected_availability
    ):
        raise RuntimeError("PIVOT feature availability drift")
    if not validated["trade_earliest_time_utc"].equals(
        expected_availability
    ):
        raise RuntimeError("PIVOT trade availability drift")
    if exact_grid:
        expected = pd.date_range(
            SOURCE_START,
            SOURCE_END,
            freq=BAR,
            inclusive="left",
        )
        if len(validated) != 420_768 or not validated["date"].equals(
            pd.Series(expected, name="date")
        ):
            raise RuntimeError("PIVOT source five-minute grid drift")
    quote = validated[
        ["spot_quote_notional", "um_quote_notional"]
    ].to_numpy(dtype=np.float64)
    signed = validated[
        ["spot_signed_quote_notional", "um_signed_quote_notional"]
    ].to_numpy(dtype=np.float64)
    numeric_valid = (
        np.isfinite(quote).all(axis=1)
        & np.isfinite(signed).all(axis=1)
        & (quote >= 0.0).all(axis=1)
        & (np.abs(signed) <= quote).all(axis=1)
    )
    validated["_row_valid"] = (
        validated["source_complete"].to_numpy(dtype=bool) & numeric_valid
    )
    return validated


def _read_source(*, nrows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(
        _path(prereg.SOURCE),
        usecols=list(prereg.SOURCE_ALLOWLIST),
        nrows=nrows,
        dtype={
            "date": "string",
            "feature_available_time_utc": "string",
            "trade_earliest_time_utc": "string",
            "spot_quote_notional": "float64",
            "um_quote_notional": "float64",
            "spot_signed_quote_notional": "float64",
            "um_signed_quote_notional": "float64",
            "source_complete": "string",
        },
    )
    frame = frame.loc[:, prereg.SOURCE_ALLOWLIST]
    return validate_source_frame(frame, exact_grid=nrows is None)


def load_source() -> pd.DataFrame:
    return _read_source()


def load_source_prefix(rows: int) -> pd.DataFrame:
    if rows <= 0 or rows >= 420_768:
        raise ValueError("PIVOT real-prefix row count must be within source")
    return _read_source(nrows=int(rows))


def swap_venues(frame: pd.DataFrame) -> pd.DataFrame:
    swapped = frame.copy()
    for left, right in (
        ("spot_quote_notional", "um_quote_notional"),
        ("spot_signed_quote_notional", "um_signed_quote_notional"),
    ):
        swapped[left], swapped[right] = frame[right].copy(), frame[left].copy()
    return swapped


def mirror_signed_flows(frame: pd.DataFrame) -> pd.DataFrame:
    mirrored = frame.copy()
    for column in (
        "spot_signed_quote_notional",
        "um_signed_quote_notional",
    ):
        mirrored[column] = -mirrored[column].to_numpy(dtype=np.float64)
    return mirrored


def _first_passage(cumulative: np.ndarray, target: float) -> int | None:
    if not np.isfinite(target) or target <= 0.0:
        return None
    indices = np.flatnonzero(cumulative >= target)
    if not len(indices):
        return None
    index = int(indices[0])
    return index if index <= LATEST_ANCHOR_INDEX else None


def _base_state_id(
    source_day: pd.Timestamp,
    spot_anchor: pd.Timestamp,
    um_anchor: pd.Timestamp,
) -> str:
    return canonical_hash(
        {
            "policy_id": prereg.Policy().policy_id,
            "source_sha256": prereg.SOURCE_SHA256,
            "source_day": _format_day(source_day),
            "spot_anchor_time": _format_time(spot_anchor),
            "um_anchor_time": _format_time(um_anchor),
        }
    )


def _aligned_day_arrays(
    frame: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray], np.ndarray]:
    first_day = frame["date"].iloc[0].floor("D")
    last_day = frame["date"].iloc[-1].floor("D")
    days = pd.date_range(first_day, last_day, freq=DAY)
    grid = pd.date_range(
        first_day,
        last_day + DAY,
        freq=BAR,
        inclusive="left",
    )
    indexed = frame.set_index("date")
    aligned = indexed.reindex(grid)
    values: dict[str, np.ndarray] = {}
    for column in (
        "spot_quote_notional",
        "um_quote_notional",
        "spot_signed_quote_notional",
        "um_signed_quote_notional",
    ):
        values[column] = (
            pd.to_numeric(aligned[column], errors="coerce")
            .to_numpy(dtype=np.float64)
            .reshape(len(days), ROWS_PER_DAY)
        )
    valid = (
        aligned["_row_valid"]
        .eq(True)
        .to_numpy(dtype=bool)
        .reshape(len(days), ROWS_PER_DAY)
    )
    return days, values, valid


def build_base_states(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    days, values, valid = _aligned_day_arrays(frame)
    spot_quote = values["spot_quote_notional"]
    um_quote = values["um_quote_notional"]
    spot_signed = values["spot_signed_quote_notional"]
    um_signed = values["um_signed_quote_notional"]
    complete = valid.all(axis=1)
    spot_totals = np.where(
        complete,
        np.sum(spot_quote, axis=1, dtype=np.float64),
        np.nan,
    )
    um_totals = np.where(
        complete,
        np.sum(um_quote, axis=1, dtype=np.float64),
        np.nan,
    )
    records: list[dict[str, Any]] = []
    funnel = {
        "calendar_days": len(days),
        "complete_reference_days": int(complete.sum()),
        "reference_ready_days": 0,
        "positive_target_days": 0,
        "both_anchor_days": 0,
        "exact_anchor_ties": 0,
        "invalid_or_missing_prefix": 0,
        "invalid_flow_denominator": 0,
        "base_paired_states": 0,
    }
    policy = prereg.Policy()

    for day_index, day in enumerate(days):
        reference_start = max(
            0,
            day_index - policy.reference_calendar_days,
        )
        spot_reference = spot_totals[reference_start:day_index]
        um_reference = um_totals[reference_start:day_index]
        spot_reference = spot_reference[np.isfinite(spot_reference)]
        um_reference = um_reference[np.isfinite(um_reference)]
        if (
            len(spot_reference) < policy.reference_complete_days_min
            or len(um_reference) < policy.reference_complete_days_min
        ):
            continue
        funnel["reference_ready_days"] += 1
        spot_expected = float(
            np.median(np.asarray(spot_reference, dtype=np.float64))
        )
        um_expected = float(
            np.median(np.asarray(um_reference, dtype=np.float64))
        )
        spot_target = policy.intrinsic_volume_fraction * spot_expected
        um_target = policy.intrinsic_volume_fraction * um_expected
        if (
            not np.isfinite(spot_target)
            or not np.isfinite(um_target)
            or spot_target <= 0.0
            or um_target <= 0.0
        ):
            continue
        funnel["positive_target_days"] += 1

        sq = np.where(np.isfinite(spot_quote[day_index]), spot_quote[day_index], 0.0)
        uq = np.where(np.isfinite(um_quote[day_index]), um_quote[day_index], 0.0)
        ss = np.where(
            np.isfinite(spot_signed[day_index]),
            spot_signed[day_index],
            0.0,
        )
        us = np.where(
            np.isfinite(um_signed[day_index]),
            um_signed[day_index],
            0.0,
        )
        cumulative_quote = {
            "spot": np.cumsum(sq, dtype=np.float64),
            "um": np.cumsum(uq, dtype=np.float64),
        }
        cumulative_signed = {
            "spot": np.cumsum(ss, dtype=np.float64),
            "um": np.cumsum(us, dtype=np.float64),
        }
        spot_anchor_index = _first_passage(
            cumulative_quote["spot"],
            spot_target,
        )
        um_anchor_index = _first_passage(
            cumulative_quote["um"],
            um_target,
        )
        if spot_anchor_index is None or um_anchor_index is None:
            continue
        funnel["both_anchor_days"] += 1
        if spot_anchor_index == um_anchor_index:
            funnel["exact_anchor_ties"] += 1
            continue
        early_index = min(spot_anchor_index, um_anchor_index)
        late_index = max(spot_anchor_index, um_anchor_index)
        buffer_index = late_index + policy.buffer_bars
        if (
            buffer_index >= ROWS_PER_DAY
            or not valid[day_index, : buffer_index + 1].all()
        ):
            funnel["invalid_or_missing_prefix"] += 1
            continue
        denominators = (
            cumulative_quote["spot"][early_index],
            cumulative_quote["um"][early_index],
            cumulative_quote["spot"][late_index],
            cumulative_quote["um"][late_index],
        )
        if not np.isfinite(denominators).all() or not all(
            float(value) > 0.0 for value in denominators
        ):
            funnel["invalid_flow_denominator"] += 1
            continue

        def flow(venue: str, index: int) -> float:
            return float(
                cumulative_signed[venue][index]
                / cumulative_quote[venue][index]
            )

        leader = "spot" if spot_anchor_index < um_anchor_index else "um"
        laggard = "um" if leader == "spot" else "spot"
        laggard_target = um_target if laggard == "um" else spot_target
        laggard_progress = float(
            cumulative_quote[laggard][early_index] / laggard_target
        )
        if (
            not np.isfinite(laggard_progress)
            or laggard_progress < 0.0
            or laggard_progress >= 1.0
        ):
            raise RuntimeError("PIVOT laggard progress invariant failed")
        day_timestamp = pd.Timestamp(day)
        spot_anchor_time = day_timestamp + spot_anchor_index * BAR
        um_anchor_time = day_timestamp + um_anchor_index * BAR
        early_time = day_timestamp + early_index * BAR
        late_time = day_timestamp + late_index * BAR
        opportunity = prereg.opportunity_times(late_time)
        spot_flow_early = flow("spot", early_index)
        um_flow_early = flow("um", early_index)
        spot_flow_late = flow("spot", late_index)
        um_flow_late = flow("um", late_index)
        raw_values = np.asarray(
            (
                laggard_progress,
                spot_flow_early,
                um_flow_early,
                spot_flow_late,
                um_flow_late,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(raw_values).all():
            raise RuntimeError("PIVOT raw state contains non-finite values")
        records.append(
            {
                "base_state_id": _base_state_id(
                    day_timestamp,
                    spot_anchor_time,
                    um_anchor_time,
                ),
                "source_day": day_timestamp,
                "spot_anchor_index": spot_anchor_index,
                "um_anchor_index": um_anchor_index,
                "early_index": early_index,
                "late_index": late_index,
                "spot_anchor_time": spot_anchor_time,
                "um_anchor_time": um_anchor_time,
                "causal_origin_time": early_time,
                "state_completion_time": opportunity["state_completion"],
                "buffer_completion_time": opportunity["buffer_completion"],
                "decision_deadline_time": opportunity["decision_deadline"],
                "entry_time": opportunity["entry"],
                "exit_time": opportunity["exit"],
                "leader_venue": leader,
                "gap_bars": late_index - early_index,
                "early_anchor_start_minute": early_index * 5,
                "laggard_progress_at_early": laggard_progress,
                "spot_flow_early": spot_flow_early,
                "um_flow_early": um_flow_early,
                "spot_flow_late": spot_flow_late,
                "um_flow_late": um_flow_late,
                "spot_abs_flow_late": abs(spot_flow_late),
                "um_abs_flow_late": abs(um_flow_late),
            }
        )
    funnel["base_paired_states"] = len(records)
    return pd.DataFrame(records, columns=BASE_COLUMNS), funnel


def prior_reference_indices(
    current_index: int,
    state_count: int,
) -> tuple[int, ...]:
    if current_index < 0 or current_index >= state_count:
        raise IndexError("PIVOT current base-state index is out of range")
    start = max(0, current_index - prereg.Policy().prior_base_states)
    return tuple(range(start, current_index))


def prior_thresholds(
    base_states: pd.DataFrame,
    current_index: int,
) -> dict[str, tuple[float, float, float]]:
    indices = prior_reference_indices(current_index, len(base_states))
    if len(indices) < prereg.Policy().prior_base_states_min:
        raise ValueError("PIVOT prior state history is not ready")
    thresholds: dict[str, tuple[float, float, float]] = {}
    for token, raw_column in QUARTILE_RAW_COLUMNS.items():
        prior = base_states.iloc[list(indices)][raw_column].to_numpy(
            dtype=np.float64
        )
        values = np.quantile(
            prior,
            np.asarray(prereg.Policy().quartiles, dtype=np.float64),
            method="linear",
        )
        if not np.isfinite(values).all():
            raise ValueError("PIVOT prior quartile threshold is non-finite")
        thresholds[token] = tuple(float(value) for value in values)
    return thresholds


def _session_token(start_minute: int) -> str:
    minute = int(start_minute)
    if minute < 0 or minute >= 24 * 60:
        raise ValueError("PIVOT early anchor minute is invalid")
    return (
        "S00_06"
        if minute < 6 * 60
        else "S06_12"
        if minute < 12 * 60
        else "S12_18"
        if minute < 18 * 60
        else "S18_24"
    )


def tokenize_base_states(
    base_states: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if list(base_states.columns) != list(BASE_COLUMNS):
        raise RuntimeError("PIVOT base-state schema drift")
    ordered = base_states.sort_values(
        ["source_day", "late_index", "base_state_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if ordered["source_day"].duplicated().any():
        raise RuntimeError("PIVOT emitted more than one base state per UTC day")
    records: list[dict[str, Any]] = []
    rejected_nonfinite = 0
    for index, row in ordered.iterrows():
        indices = prior_reference_indices(index, len(ordered))
        if len(indices) < prereg.Policy().prior_base_states_min:
            continue
        prior = ordered.iloc[list(indices)]
        previous = ordered.iloc[index - 1]
        try:
            tokens = {
                "leader": str(row["leader_venue"]).upper(),
                "gap_q": prereg.prior_quartile_bucket(
                    float(row["gap_bars"]),
                    prior["gap_bars"].to_numpy(dtype=np.float64),
                ),
                "early_session": _session_token(
                    int(row["early_anchor_start_minute"])
                ),
                "laggard_progress_q": prereg.prior_quartile_bucket(
                    float(row["laggard_progress_at_early"]),
                    prior["laggard_progress_at_early"].to_numpy(
                        dtype=np.float64
                    ),
                ),
                "spot_early_sign": prereg.sign_token(
                    float(row["spot_flow_early"])
                ),
                "um_early_sign": prereg.sign_token(
                    float(row["um_flow_early"])
                ),
                "spot_late_sign": prereg.sign_token(
                    float(row["spot_flow_late"])
                ),
                "um_late_sign": prereg.sign_token(
                    float(row["um_flow_late"])
                ),
                "spot_late_abs_flow_q": prereg.prior_quartile_bucket(
                    float(row["spot_abs_flow_late"]),
                    prior["spot_abs_flow_late"].to_numpy(dtype=np.float64),
                ),
                "um_late_abs_flow_q": prereg.prior_quartile_bucket(
                    float(row["um_abs_flow_late"]),
                    prior["um_abs_flow_late"].to_numpy(dtype=np.float64),
                ),
                "gap_change": (
                    "NARROW"
                    if int(row["gap_bars"]) < int(previous["gap_bars"])
                    else "WIDEN"
                    if int(row["gap_bars"]) > int(previous["gap_bars"])
                    else "SAME"
                ),
                "leader_change": (
                    "SAME"
                    if str(row["leader_venue"])
                    == str(previous["leader_venue"])
                    else "SWITCH"
                ),
            }
            tokens = prereg.validate_tokens(tokens)
        except ValueError:
            rejected_nonfinite += 1
            continue
        record = {column: row[column] for column in BASE_COLUMNS}
        record.update(tokens)
        records.append(record)
    columns = (*BASE_COLUMNS, *prereg.TOKEN_COLUMNS)
    return (
        pd.DataFrame(records, columns=columns),
        {
            "prior_history_ready_states": max(
                0,
                len(ordered) - prereg.Policy().prior_base_states_min,
            ),
            "token_ready_states": len(records),
            "tokenization_rejections": rejected_nonfinite,
        },
    )


def serialize_token_state(tokens: Mapping[str, str]) -> str:
    normalized = prereg.validate_tokens(tokens)
    return "\n".join(
        f"{name}={normalized[name]}" for name in prereg.TOKEN_COLUMNS
    )


def reservation_mask(states: pd.DataFrame) -> pd.Series:
    required = {"entry_time", "exit_time", "base_state_id"}
    if not required.issubset(states.columns):
        raise RuntimeError("PIVOT reservation input is missing clock columns")
    selected = pd.Series(False, index=states.index, dtype=bool)
    previous_exit: pd.Timestamp | None = None
    ordered = states.sort_values(
        ["entry_time", "base_state_id"],
        kind="mergesort",
    )
    for row in ordered.itertuples():
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if exit_time <= entry:
            raise RuntimeError("PIVOT reservation interval is invalid")
        if previous_exit is not None and entry < previous_exit:
            continue
        selected.at[row.Index] = True
        previous_exit = exit_time
    return selected


def _contained_in_split(row: Any, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    values = (
        pd.Timestamp(row.source_day),
        pd.Timestamp(row.spot_anchor_time),
        pd.Timestamp(row.um_anchor_time),
        pd.Timestamp(row.causal_origin_time),
        pd.Timestamp(row.state_completion_time),
        pd.Timestamp(row.buffer_completion_time),
        pd.Timestamp(row.decision_deadline_time),
        pd.Timestamp(row.entry_time),
        pd.Timestamp(row.exit_time),
    )
    return bool(all(start <= value < end for value in values))


def assign_primary_reservation(
    token_states: pd.DataFrame,
) -> pd.DataFrame:
    columns = (*BASE_COLUMNS, *prereg.TOKEN_COLUMNS)
    if list(token_states.columns) != list(columns):
        raise RuntimeError("PIVOT token-state schema drift")
    clock = token_states.copy()
    clock["primary_reserved"] = reservation_mask(clock)
    split_values: list[str] = []
    for row in clock.itertuples(index=False):
        if not bool(row.primary_reserved):
            split_values.append("")
            continue
        matches = [
            name
            for name, (start, end) in SPLITS.items()
            if _contained_in_split(row, start, end)
        ]
        if len(matches) > 1:
            raise RuntimeError("PIVOT state belongs to multiple temporal splits")
        split_values.append(matches[0] if matches else "")
    clock["primary_split"] = split_values
    return clock.loc[:, CLOCK_COLUMNS]


def build_state_clock(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    base, base_funnel = build_base_states(frame)
    token_states, token_funnel = tokenize_base_states(base)
    clock = assign_primary_reservation(token_states)
    funnel = {
        **base_funnel,
        **token_funnel,
        "globally_reserved_states": int(clock["primary_reserved"].sum()),
        "split_contained_states": int(clock["primary_split"].ne("").sum()),
        "split_crossing_reserved_states": int(
            (
                clock["primary_reserved"]
                & clock["primary_split"].eq("")
            ).sum()
        ),
    }
    return clock, funnel


def _selected(clock: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    selected = clock.loc[clock["primary_reserved"]]
    if split is None:
        selected = selected.loc[selected["primary_split"].ne("")]
    else:
        selected = selected.loc[selected["primary_split"].eq(split)]
    return selected.sort_values(
        ["entry_time", "base_state_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _period_counts(
    rows: pd.DataFrame,
    frequency: str,
) -> dict[str, int]:
    if rows.empty:
        return {}
    naive = rows["entry_time"].dt.tz_convert(None)
    values = naive.dt.to_period(frequency).astype(str).value_counts()
    return {str(key): int(value) for key, value in sorted(values.items())}


def temporal_stats(rows: pd.DataFrame) -> dict[str, Any]:
    ordered = rows.sort_values("entry_time", kind="mergesort")
    count = len(ordered)
    month_counts = _period_counts(ordered, "M")
    quarter_counts = _period_counts(ordered, "Q")
    gaps = ordered["entry_time"].diff().dropna()
    return {
        "opportunities": count,
        "active_months": len(month_counts),
        "month_counts": month_counts,
        "quarter_counts": quarter_counts,
        "maximum_month_share": (
            float(max(month_counts.values()) / count) if count else None
        ),
        "maximum_entry_gap_days": (
            float(gaps.max() / DAY) if not gaps.empty else None
        ),
    }


def token_stats(rows: pd.DataFrame) -> dict[str, Any]:
    total = len(rows)
    levels: dict[str, Any] = {}
    for name, vocabulary in prereg.TOKEN_SCHEMA:
        counts = rows[name].value_counts().to_dict() if total else {}
        levels[name] = {
            "counts": {
                level: int(counts.get(level, 0)) for level in vocabulary
            },
            "shares": {
                level: (
                    float(counts.get(level, 0) / total) if total else 0.0
                )
                for level in vocabulary
            },
            "observed": [
                level for level in vocabulary if counts.get(level, 0)
            ],
        }
    if total:
        signatures = (
            rows.loc[:, prereg.TOKEN_COLUMNS]
            .astype(str)
            .agg("|".join, axis=1)
            .value_counts()
        )
        maximum_signature_share = float(signatures.max() / total)
        unique_signatures = int(len(signatures))
    else:
        maximum_signature_share = None
        unique_signatures = 0
    invalid = 0
    for name, vocabulary in prereg.TOKEN_SCHEMA:
        invalid += int((~rows[name].isin(vocabulary)).sum())
    return {
        "opportunities": total,
        "levels": levels,
        "unique_signatures": unique_signatures,
        "maximum_exact_signature_share": maximum_signature_share,
        "invalid_or_missing_tokens": invalid
        + int(rows.loc[:, prereg.TOKEN_COLUMNS].isna().sum().sum()),
    }


def _year_rows(rows: pd.DataFrame, year: int) -> pd.DataFrame:
    return rows.loc[rows["entry_time"].dt.year.eq(year)].copy()


def _half_counts(rows: pd.DataFrame, year: int) -> tuple[int, int]:
    first = pd.Timestamp(f"{year}-01-01T00:00:00Z")
    middle = pd.Timestamp(f"{year}-07-01T00:00:00Z")
    end = pd.Timestamp(f"{year + 1}-01-01T00:00:00Z")
    return (
        int(rows["entry_time"].ge(first).mul(rows["entry_time"].lt(middle)).sum()),
        int(rows["entry_time"].ge(middle).mul(rows["entry_time"].lt(end)).sum()),
    )


def _quarter_counts(rows: pd.DataFrame, year: int) -> tuple[int, ...]:
    boundaries = [
        pd.Timestamp(f"{year}-{month:02d}-01T00:00:00Z")
        for month in (1, 4, 7, 10)
    ] + [pd.Timestamp(f"{year + 1}-01-01T00:00:00Z")]
    return tuple(
        int(
            rows["entry_time"]
            .ge(boundaries[index])
            .mul(rows["entry_time"].lt(boundaries[index + 1]))
            .sum()
        )
        for index in range(4)
    )


def _token_checks(
    name: str,
    report: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    levels = report["levels"]
    checks: dict[str, bool] = {}
    for level in ("SPOT", "UM"):
        checks[f"{name}:leader:{level}"] = (
            levels["leader"]["shares"][level]
            >= gates["each_leader_share_min"]
        )
    for token in prereg.SIGN_TOKEN_COLUMNS:
        for level in ("NEG", "POS"):
            checks[f"{name}:{token}:{level}"] = (
                levels[token]["shares"][level]
                >= gates["each_sign_negative_positive_share_min"]
            )
    minimum, maximum = gates["each_quartile_share_range"]
    for token in QUARTILE_RAW_COLUMNS:
        for level in ("Q0", "Q1", "Q2", "Q3"):
            share = levels[token]["shares"][level]
            checks[f"{name}:{token}:{level}"] = minimum <= share <= maximum
    session = levels["early_session"]
    checks[f"{name}:session_levels"] = (
        len(session["observed"]) >= gates["session_levels_min"]
    )
    checks[f"{name}:session_max_share"] = (
        max(session["shares"].values(), default=0.0)
        <= gates["max_session_share"]
    )
    for level in ("NARROW", "WIDEN"):
        checks[f"{name}:gap_change:{level}"] = (
            levels["gap_change"]["shares"][level]
            >= gates["gap_narrow_widen_each_share_min"]
        )
    checks[f"{name}:leader_change:SWITCH"] = (
        levels["leader_change"]["shares"]["SWITCH"]
        >= gates["leader_switch_share_min"]
    )
    signature_share = report["maximum_exact_signature_share"]
    checks[f"{name}:maximum_exact_signature_share"] = bool(
        signature_share is not None
        and signature_share <= gates["max_exact_signature_share"]
    )
    checks[f"{name}:valid_tokens"] = (
        report["invalid_or_missing_tokens"] == 0
    )
    return checks


def support_checks(
    clock: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    gates = prereg.build_manifest()["source_support_gates"]
    selected = {
        "global": _selected(clock),
        "train": _selected(clock, "train"),
        "selection": _selected(clock, "selection"),
        "eval": _selected(clock, "eval"),
    }
    temporal = {name: temporal_stats(rows) for name, rows in selected.items()}
    tokens = {
        name: token_stats(selected[name])
        for name in ("train", "selection", "eval")
    }
    checks: dict[str, bool] = {
        "global_opportunities": (
            temporal["global"]["opportunities"]
            >= gates["global_opportunities_min"]
        ),
        "train_opportunities": (
            temporal["train"]["opportunities"]
            >= gates["train_opportunities_min"]
        ),
    }
    train_2020 = _year_rows(selected["train"], 2020)
    train_2021 = _year_rows(selected["train"], 2021)
    checks["train_2020_opportunities"] = (
        len(train_2020) >= gates["each_train_year_min"]
    )
    checks["train_2021_opportunities"] = (
        len(train_2021) >= gates["each_train_year_min"]
    )
    checks["train_2020_active_months"] = (
        temporal_stats(train_2020)["active_months"]
        >= gates["train_2020_active_months_min"]
    )
    checks["train_2021_active_months"] = (
        temporal_stats(train_2021)["active_months"]
        >= gates["train_2021_active_months_min"]
    )
    checks["train_maximum_month_share"] = bool(
        temporal["train"]["maximum_month_share"] is not None
        and temporal["train"]["maximum_month_share"]
        <= gates["max_month_share"]
    )
    checks["train_maximum_entry_gap"] = bool(
        temporal["train"]["maximum_entry_gap_days"] is not None
        and temporal["train"]["maximum_entry_gap_days"]
        <= gates["max_entry_gap_days"]
    )

    for split, year, minimum_key, month_key in (
        (
            "selection",
            2022,
            "selection_opportunities_min",
            "selection_2022_active_months_min",
        ),
        (
            "eval",
            2023,
            "eval_opportunities_min",
            "eval_2023_active_months_min",
        ),
    ):
        rows = selected[split]
        halves = _half_counts(rows, year)
        quarters = _quarter_counts(rows, year)
        checks[f"{split}_opportunities"] = (
            len(rows) >= gates[minimum_key]
        )
        checks[f"{split}_each_half"] = all(
            count >= gates["selection_eval_each_half_min"]
            for count in halves
        )
        checks[f"{split}_each_quarter"] = all(
            count >= gates["selection_eval_each_quarter_min"]
            for count in quarters
        )
        checks[f"{split}_active_months"] = (
            temporal[split]["active_months"] >= gates[month_key]
        )
        checks[f"{split}_maximum_month_share"] = bool(
            temporal[split]["maximum_month_share"] is not None
            and temporal[split]["maximum_month_share"]
            <= gates["max_month_share"]
        )
        checks[f"{split}_maximum_entry_gap"] = bool(
            temporal[split]["maximum_entry_gap_days"] is not None
            and temporal[split]["maximum_entry_gap_days"]
            <= gates["max_entry_gap_days"]
        )
        temporal[split]["half_counts"] = list(halves)
        temporal[split]["quarter_gate_counts"] = list(quarters)

    for name, report in tokens.items():
        checks.update(_token_checks(name, report, gates))

    train_levels = {
        token: set(tokens["train"]["levels"][token]["observed"])
        for token in prereg.TOKEN_COLUMNS
    }
    for downstream in ("selection", "eval"):
        checks[f"{downstream}:levels_exist_in_train"] = all(
            set(tokens[downstream]["levels"][token]["observed"]).issubset(
                train_levels[token]
            )
            for token in prereg.TOKEN_COLUMNS
        )
    return temporal, tokens, checks


def deterministic_clock_bytes(clock: pd.DataFrame) -> bytes:
    if list(clock.columns) != list(CLOCK_COLUMNS):
        raise RuntimeError("PIVOT clock schema drift")
    if any(
        token in column.lower()
        for column in CLOCK_COLUMNS
        for token in FORBIDDEN_CLOCK_TOKENS
    ):
        raise RuntimeError("PIVOT clock contains a forbidden outcome field")
    serialized = clock.copy()
    serialized["source_day"] = serialized["source_day"].map(_format_day)
    for column in TIME_COLUMNS:
        serialized[column] = serialized[column].map(_format_time)
    serialized["primary_reserved"] = serialized["primary_reserved"].map(
        {True: "true", False: "false"}
    )
    text = serialized.to_csv(
        index=False,
        columns=CLOCK_COLUMNS,
        lineterminator="\n",
        float_format="%.17g",
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


def _core_payload(
    clock: pd.DataFrame,
    funnel: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    clock_bytes: bytes,
    *,
    artifact_eligible: bool,
) -> dict[str, Any]:
    temporal, tokens, checks = support_checks(clock)
    source_support_passed = bool(
        artifact_eligible and checks and all(checks.values())
    )
    first_failure = next(
        (name for name, passed in checks.items() if not passed),
        None,
    )
    if not artifact_eligible:
        decision = "synthetic_build_cannot_authorize_next_stage"
    elif not source_support_passed:
        decision = "retire_PIVOT_72_unchanged_before_outcomes"
    else:
        decision = "advance_to_frozen_cheap_causal_baseline_evaluator"
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.Policy().policy_id,
        "artifact_eligible": artifact_eligible,
        "outcomes_opened": False,
        "market_values_loaded": False,
        "funding_values_loaded": False,
        "comparator_rows_decoded": 0,
        "post_2023_values_decoded": False,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "implementation": {
            "source": str(SCRIPT_PATH),
            "source_sha256": sha256_file(SCRIPT_PATH),
            "tests": str(TEST_PATH),
            "tests_sha256": sha256_file(TEST_PATH),
        },
        "source_audit": dict(source_audit),
        "feature_funnel": dict(funnel),
        "temporal_statistics": temporal,
        "token_statistics": tokens,
        "source_support_checks": checks,
        "source_support_passed": source_support_passed,
        "first_failing_check": first_failure,
        "required_builder_tests": preregistration[
            "source_support_gates"
        ]["required_builder_tests"],
        "clock": {
            "path": str(DEFAULT_CLOCK_OUTPUT),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": len(clock),
            "globally_reserved_rows": int(
                clock["primary_reserved"].sum()
            ),
            "split_contained_rows": int(clock["primary_split"].ne("").sum()),
            "columns": list(CLOCK_COLUMNS),
        },
        "decision": decision,
        "authorized_next_stage": (
            "freeze_cheap_causal_baseline_evaluator"
            if source_support_passed
            else None
        ),
        "outcome_boundary": {
            "source_rows_decoded": int(
                source_audit.get("source_rows_decoded", 0)
            ),
            "market_rows_decoded": 0,
            "funding_rows_decoded": 0,
            "comparator_rows_decoded": 0,
            "post_entry_price_rows_decoded": 0,
            "future_return_rows_decoded": 0,
            "return_or_pnl_fields_decoded": 0,
            "network_calls": 0,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def build_support_from_frame(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], bytes]:
    clock, funnel = build_state_clock(frame)
    clock_bytes = deterministic_clock_bytes(clock)
    report = _core_payload(
        clock,
        funnel,
        {
            "source_rows_decoded": 0,
            "synthetic_or_injected": True,
        },
        validate_preregistration(),
        clock_bytes,
        artifact_eligible=False,
    )
    return report, clock_bytes


def build_real_support_payload() -> tuple[dict[str, Any], bytes]:
    _assert_protocol_committed()
    preregistration = validate_preregistration()
    bindings = verify_pre_source_bindings()
    source = load_source()
    clock, funnel = build_state_clock(source)
    clock_bytes = deterministic_clock_bytes(clock)
    source_audit = {
        "path": prereg.SOURCE,
        "sha256": prereg.SOURCE_SHA256,
        "header_sha256": prereg.SOURCE_HEADER_SHA256,
        "source_rows_decoded": len(source),
        "allowlist": list(prereg.SOURCE_ALLOWLIST),
        "first_timestamp": _format_time(source["date"].iloc[0]),
        "last_timestamp": _format_time(source["date"].iloc[-1]),
        "valid_rows": int(source["_row_valid"].sum()),
        "invalid_rows": int((~source["_row_valid"]).sum()),
        "pre_source_bindings": bindings,
        "synthetic_or_injected": False,
    }
    return (
        _core_payload(
            clock,
            funnel,
            source_audit,
            preregistration,
            clock_bytes,
            artifact_eligible=True,
        ),
        clock_bytes,
    )


def _write_once(path: str | Path, payload: bytes) -> str:
    output = _path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError(f"PIVOT noncanonical existing artifact: {path}")
        return "verified_existing"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            if output.read_bytes() != payload:
                raise RuntimeError(f"PIVOT artifact race drift: {path}")
            return "verified_existing"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def write_support(
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> dict[str, Any]:
    if Path(report_output) != DEFAULT_REPORT_OUTPUT:
        raise RuntimeError("PIVOT real report output path is frozen")
    if Path(clock_output) != DEFAULT_CLOCK_OUTPUT:
        raise RuntimeError("PIVOT real clock output path is frozen")
    report, clock_bytes = build_real_support_payload()
    clock_status = _write_once(clock_output, clock_bytes)
    report_bytes = (
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    report_status = _write_once(report_output, report_bytes)
    return {
        "report_status": report_status,
        "clock_status": clock_status,
        "report": str(report_output),
        "clock": str(clock_output),
        "source_support_passed": report["source_support_passed"],
        "decision": report["decision"],
        "manifest_hash": report["manifest_hash"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_REPORT_OUTPUT))
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK_OUTPUT))
    args = parser.parse_args()
    print(
        json.dumps(
            write_support(args.output, args.clock_output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
