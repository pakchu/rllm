"""Build outcome-blind BFWC-288 source support and prior-clock novelty."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from training import preregister_blockspace_fee_witness_concordance as prereg


PROTOCOL_VERSION = "blockspace_fee_witness_concordance_support_v1"
POLICY_ID = "BFWC-288"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_blockspace_fee_witness_concordance_support.py")
TEST_PATHS = (
    Path("tests/test_build_blockspace_fee_witness_concordance_support.py"),
    Path(
        "tests/test_blockspace_fee_witness_concordance_"
        "preregistration_artifact.py"
    ),
)
PREREGISTRATION = Path(
    "results/blockspace_fee_witness_concordance_preregistration_2026-07-30.json"
)
PREREGISTRATION_SHA256 = (
    "c255cccbda22cdc8c43e35f04f5d1792f0a76f88caa966434b5be79bff1f65f7"
)
PREREGISTRATION_MANIFEST_HASH = (
    "499bdcd199bfe8ae7dad9bf5e51271f8fb1fd762edbaf4a1e0d026708e9fdf9b"
)

DEFAULT_REPORT_OUTPUT = Path(
    "results/blockspace_fee_witness_concordance_support_2026-07-30.json"
)
DEFAULT_PRIMARY_CLOCK_OUTPUT = Path(
    "results/blockspace_fee_witness_concordance_primary_clock_2026-07-30.csv.gz"
)
DEFAULT_CONTROL_CLOCK_OUTPUT = Path(
    "results/blockspace_fee_witness_concordance_control_clocks_2026-07-30.csv.gz"
)

BAR = pd.Timedelta(minutes=5)
BUCKET = pd.Timedelta(hours=12)
HOLD = pd.Timedelta(hours=24)
FULL_START = pd.Timestamp("2023-06-01T00:00:00Z")
FULL_END = pd.Timestamp("2026-06-01T00:00:00Z")
SPLITS = {
    "selection": (
        FULL_START,
        pd.Timestamp("2025-01-01T00:00:00Z"),
    ),
    "future_2025": (
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:00:00Z"),
    ),
    "future_2026": (
        pd.Timestamp("2026-01-01T00:00:00Z"),
        FULL_END,
    ),
}

BFRT_USECOLS = (
    "bucket_start_utc",
    "bucket_end_utc",
    "available_at_utc",
    "fee_p10",
    "fee_p25",
    "fee_p75",
    "fee_p90",
)
WCTR_USECOLS = (
    "bucket_start_utc",
    "bucket_end_utc",
    "available_at_utc",
    "avg_size",
    "avg_weight",
)
KEY_COLUMNS = ("bucket_start_utc", "bucket_end_utc")
FEATURE_COLUMNS = (
    "bucket_start_utc",
    "bucket_end_utc",
    "joint_available_at_utc",
    "base_valid",
    "R",
    "W",
    "U",
    "Q",
    "rank_L",
    "rank_E",
    "rank_n",
    "rank",
    "q_rank_L",
    "q_rank_E",
    "q_rank_n",
    "q_rank",
)
CLOCK_COLUMNS = (
    "policy_id",
    "control",
    "window",
    "signal_id",
    "bucket_start_utc",
    "bucket_end_utc",
    "source_available_at_utc",
    "entry_time_utc",
    "exit_time_utc",
    "side",
    "rank_L",
    "rank_E",
    "rank_n",
    "rank",
)
INDEPENDENT_CONTROLS = (
    "primary",
    "fee_rotation_only",
    "witness_fullness_only",
    "drop_witness",
    "drop_fullness",
    "one_bucket_stale_witness_fullness",
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


def validate_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("BFWC-288 preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_manifest(payload)
    if payload != prereg.build_manifest():
        raise RuntimeError("BFWC-288 preregistration differs from build_manifest")
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("BFWC-288 preregistration manifest_hash drift")
    for field in (
        "source_rows_opened",
        "source_incidence_opened",
        "candidate_overlap_opened",
        "economic_rows_opened",
        "outcomes_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"BFWC-288 preregistration boundary opened: {field}")
    return payload


def _assert_protocol_committed() -> None:
    paths = [str(SCRIPT_PATH), *(str(path) for path in TEST_PATHS)]
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *paths],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if tracked.returncode:
        raise RuntimeError("BFWC-288 source-support evaluator is not committed")
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *paths],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if clean.returncode:
        raise RuntimeError("BFWC-288 source-support evaluator differs from HEAD")


def _format_time(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("BFWC-288 timestamps must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.microsecond or timestamp.nanosecond:
        raise RuntimeError("BFWC-288 timestamps must be whole-second")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time_column(series: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="raise")
    if parsed.isna().any():
        raise RuntimeError(f"BFWC-288 null timestamp: {name}")
    if any(value.microsecond or value.nanosecond for value in parsed):
        raise RuntimeError(f"BFWC-288 subsecond timestamp drift: {name}")
    return parsed


def _reject_forbidden_columns(columns: Iterable[str]) -> None:
    for column in columns:
        lowered = str(column).lower()
        if any(token in lowered for token in FORBIDDEN_TOKENS):
            raise RuntimeError(f"BFWC-288 outcome-like column rejected: {column}")


def validate_source_frame(
    frame: pd.DataFrame,
    source: str,
    *,
    exact_domain: bool = True,
) -> pd.DataFrame:
    if source not in {"BFRT", "WCTR"}:
        raise ValueError("BFWC-288 source must be BFRT or WCTR")
    expected = BFRT_USECOLS if source == "BFRT" else WCTR_USECOLS
    if list(frame.columns) != list(expected):
        raise RuntimeError(f"BFWC-288 {source} exact usecols/schema drift")
    _reject_forbidden_columns(frame.columns)
    validated = frame.copy()
    for column in (*KEY_COLUMNS, "available_at_utc"):
        validated[column] = _parse_time_column(validated[column], column)
    numeric = expected[3:]
    for column in numeric:
        validated[column] = pd.to_numeric(
            validated[column], errors="coerce"
        ).astype(np.float64)
    if validated.empty:
        raise RuntimeError(f"BFWC-288 {source} source is empty")
    if validated[list(KEY_COLUMNS)].duplicated().any():
        raise RuntimeError(f"BFWC-288 {source} duplicate join key")
    if not validated["bucket_start_utc"].is_monotonic_increasing:
        raise RuntimeError(f"BFWC-288 {source} time order drift")
    if not (
        validated["bucket_end_utc"] - validated["bucket_start_utc"]
    ).eq(BUCKET).all():
        raise RuntimeError(f"BFWC-288 {source} bucket duration drift")
    epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    if not (
        (validated["bucket_start_utc"] - epoch).map(lambda value: value % BUCKET)
        == pd.Timedelta(0)
    ).all():
        raise RuntimeError(f"BFWC-288 {source} bucket grid drift")
    expected_available = validated["bucket_end_utc"] + pd.Timedelta(hours=48)
    if not validated["available_at_utc"].equals(expected_available):
        raise RuntimeError(f"BFWC-288 {source} availability clock drift")
    if not validated["available_at_utc"].is_monotonic_increasing:
        raise RuntimeError(f"BFWC-288 {source} availability order drift")
    values = validated[list(numeric)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"BFWC-288 {source} nonfinite primitive")
    if source == "BFRT":
        if (values < 0.0).any():
            raise RuntimeError("BFWC-288 fee percentile domain drift")
        if not np.all(values[:, 1:] >= values[:, :-1]):
            raise RuntimeError("BFWC-288 fee percentile ordering drift")
    else:
        size = validated["avg_size"].to_numpy(dtype=np.float64)
        weight = validated["avg_weight"].to_numpy(dtype=np.float64)
        if (size <= 0.0).any() or (weight < 0.0).any() or (weight > 4_000_000).any():
            raise RuntimeError("BFWC-288 witness primitive domain drift")
        share = (4.0 * size - weight) / (3.0 * size)
        if (
            not np.isfinite(share).all()
            or (share < 0.0).any()
            or (share > 1.0).any()
        ):
            raise RuntimeError("BFWC-288 witness_share domain drift")
    if exact_domain:
        deltas = validated["bucket_start_utc"].diff().dropna()
        if not deltas.eq(BUCKET).all():
            raise RuntimeError(f"BFWC-288 {source} source gap drift")
    return validated


def _validate_exact_header(path: str | Path, expected: str) -> None:
    if prereg.csv_header_bytes(path) != expected.encode("utf-8"):
        raise RuntimeError(f"BFWC-288 exact CSV header drift: {path}")


def load_source(
    path: str | Path,
    source: str,
) -> pd.DataFrame:
    expected_path = prereg.BFRT_NORMALIZED if source == "BFRT" else prereg.WCTR_NORMALIZED
    if Path(path) != expected_path:
        raise RuntimeError(f"BFWC-288 {source} source path drift")
    expected_header = prereg.EXACT_CSV_HEADERS[str(expected_path)]
    _validate_exact_header(path, expected_header)
    expected_hash = prereg.FROZEN_DEPENDENCIES[str(expected_path)]
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f"BFWC-288 {source} source hash drift")
    usecols = BFRT_USECOLS if source == "BFRT" else WCTR_USECOLS
    frame = pd.read_csv(
        _path(path),
        usecols=list(usecols),
        dtype={column: "string" for column in usecols},
    )
    frame = frame.loc[:, usecols]
    return validate_source_frame(frame, source)


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        load_source(prereg.BFRT_NORMALIZED, "BFRT"),
        load_source(prereg.WCTR_NORMALIZED, "WCTR"),
    )


def validate_source_bindings() -> dict[str, Any]:
    """Validate only the two source lineages before source rows are decoded."""

    audit: dict[str, Any] = {}
    contracts = (
        (
            "BFRT",
            prereg.BFRT_SOURCE_MANIFEST,
            prereg.BFRT_SOURCE_MANIFEST_HASH,
            prereg.BFRT_NORMALIZED,
        ),
        (
            "WCTR",
            prereg.WCTR_SOURCE_MANIFEST,
            prereg.WCTR_SOURCE_MANIFEST_HASH,
            prereg.WCTR_NORMALIZED,
        ),
    )
    for source, path, internal_hash, normalized in contracts:
        expected_file_hash = prereg.FROZEN_DEPENDENCIES[str(path)]
        actual_file_hash = sha256_file(path)
        if actual_file_hash != expected_file_hash:
            raise RuntimeError(f"BFWC-288 {source} source manifest hash drift")
        prereg._validate_source_manifest(path, internal_hash)
        normalized_hash = sha256_file(normalized)
        expected_normalized_hash = prereg.FROZEN_DEPENDENCIES[str(normalized)]
        if normalized_hash != expected_normalized_hash:
            raise RuntimeError(f"BFWC-288 {source} normalized source hash drift")
        expected_header = prereg.EXACT_CSV_HEADERS[str(normalized)]
        _validate_exact_header(normalized, expected_header)
        audit[source] = {
            "source_manifest_path": str(path),
            "source_manifest_sha256": actual_file_hash,
            "source_manifest_hash": internal_hash,
            "normalized_path": str(normalized),
            "normalized_sha256": normalized_hash,
            "normalized_header_sha256": hashlib.sha256(
                expected_header.encode("utf-8")
            ).hexdigest(),
        }
    return audit


def _midrank_parts(current: float, prior: np.ndarray) -> tuple[int, int, int, float]:
    if not math.isfinite(current) or not len(prior) or not np.isfinite(prior).all():
        raise RuntimeError("BFWC-288 rank requires finite current and history")
    lower = int(np.sum(prior < current))
    equal = int(np.sum(prior == current))
    count = int(len(prior))
    return lower, equal, count, float((lower + 0.5 * equal) / count)


def _rank_feature(
    values: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.full(len(values), -1, dtype=np.int64)
    equal = np.full(len(values), -1, dtype=np.int64)
    count = np.zeros(len(values), dtype=np.int64)
    rank = np.full(len(values), np.nan, dtype=np.float64)
    valid_indices: list[int] = []
    for index in range(len(values)):
        if not valid[index]:
            continue
        history_indices = valid_indices[-180:]
        if history_indices:
            parts = _midrank_parts(values[index], values[history_indices])
            lower[index], equal[index], count[index], rank[index] = parts
        valid_indices.append(index)
    return lower, equal, count, rank


def build_joint_features(
    bfrt: pd.DataFrame,
    wctr: pd.DataFrame,
    *,
    exact_domain: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = validate_source_frame(bfrt, "BFRT", exact_domain=exact_domain)
    right = validate_source_frame(wctr, "WCTR", exact_domain=exact_domain)
    common_start = max(
        left["bucket_start_utc"].iloc[0],
        right["bucket_start_utc"].iloc[0],
        FULL_START,
    )
    common_last = min(
        left["bucket_start_utc"].iloc[-1],
        right["bucket_start_utc"].iloc[-1],
        FULL_END - BUCKET,
    )
    if common_start > common_last:
        raise RuntimeError("BFWC-288 empty common source domain")
    left = left.loc[
        left["bucket_start_utc"].between(common_start, common_last)
    ].reset_index(drop=True)
    right = right.loc[
        right["bucket_start_utc"].between(common_start, common_last)
    ].reset_index(drop=True)
    left_keys = pd.MultiIndex.from_frame(left[list(KEY_COLUMNS)])
    right_keys = pd.MultiIndex.from_frame(right[list(KEY_COLUMNS)])
    gaps = len(left_keys.symmetric_difference(right_keys))
    if gaps != 0:
        raise RuntimeError("BFWC-288 exact join gap drift")
    joined = left.merge(
        right,
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
        suffixes=("_bfrt", "_wctr"),
        sort=True,
    )
    if len(joined) != len(left) or len(joined) != len(right):
        raise RuntimeError("BFWC-288 exact join cardinality drift")
    starts = joined["bucket_start_utc"]
    consecutive = (
        starts.diff().eq(BUCKET) & starts.diff(2).eq(2 * BUCKET)
    ).to_numpy(dtype=bool)
    fees = {
        percentile: np.log1p(
            joined[f"fee_p{percentile}"].to_numpy(dtype=np.float64)
        )
        for percentile in (10, 25, 75, 90)
    }
    delta = {key: value - np.roll(value, 2) for key, value in fees.items()}
    rotation = 0.5 * (
        (delta[10] + delta[25]) - (delta[75] + delta[90])
    )
    size = joined["avg_size"].to_numpy(dtype=np.float64)
    weight = joined["avg_weight"].to_numpy(dtype=np.float64)
    witness_share = (4.0 * size - weight) / (3.0 * size)
    fullness = weight / 4_000_000.0
    witness = witness_share - np.roll(witness_share, 2)
    full = fullness - np.roll(fullness, 2)
    rotation[~consecutive] = np.nan
    witness[~consecutive] = np.nan
    full[~consecutive] = np.nan
    base_valid = consecutive & np.isfinite(rotation) & np.isfinite(witness) & np.isfinite(full)
    q_value = 0.5 * (np.abs(witness) + np.abs(full))
    rank_l, rank_e, rank_n, rank = _rank_feature(
        np.abs(rotation), base_valid
    )
    q_l, q_e, q_n, q_rank = _rank_feature(q_value, base_valid)
    features = pd.DataFrame(
        {
            "bucket_start_utc": joined["bucket_start_utc"],
            "bucket_end_utc": joined["bucket_end_utc"],
            "joint_available_at_utc": pd.concat(
                [
                    joined["available_at_utc_bfrt"],
                    joined["available_at_utc_wctr"],
                ],
                axis=1,
            ).max(axis=1),
            "base_valid": base_valid,
            "R": rotation,
            "W": witness,
            "U": full,
            "Q": q_value,
            "rank_L": rank_l,
            "rank_E": rank_e,
            "rank_n": rank_n,
            "rank": rank,
            "q_rank_L": q_l,
            "q_rank_E": q_e,
            "q_rank_n": q_n,
            "q_rank": q_rank,
        },
        columns=FEATURE_COLUMNS,
    )
    audit = {
        "common_start": _format_time(common_start),
        "common_last_bucket_start": _format_time(common_last),
        "BFRT_rows_in_common_domain": len(left),
        "WCTR_rows_in_common_domain": len(right),
        "joined_rows": len(joined),
        "exact_join_gaps": gaps,
        "base_valid_rows": int(features["base_valid"].sum()),
    }
    return features, audit


def _side(value: float) -> str:
    if value > 0.0:
        return "LONG"
    if value < 0.0:
        return "SHORT"
    raise RuntimeError("BFWC-288 zero has no side")


def signal_id(row: Mapping[str, Any] | Any, side: str) -> str:
    def value(name: str) -> Any:
        return row[name] if isinstance(row, Mapping) else getattr(row, name)

    token = "|".join(
        (
            POLICY_ID,
            _format_time(value("bucket_start_utc")),
            _format_time(value("bucket_end_utc")),
            _format_time(value("joint_available_at_utc")),
            side,
        )
    )
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _entry_time(availability: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(availability)
    epoch_seconds = int(timestamp.timestamp())
    return pd.Timestamp(
        prereg.ceil_5m_plus_one_bar(epoch_seconds),
        unit="s",
        tz="UTC",
    )


def _raw_candidate(
    row: Any,
    control: str,
    side: str,
    *,
    use_q_rank: bool = False,
) -> dict[str, Any]:
    entry = _entry_time(row.joint_available_at_utc)
    return {
        "policy_id": POLICY_ID,
        "control": control,
        "window": None,
        "signal_id": signal_id(row, side),
        "bucket_start_utc": row.bucket_start_utc,
        "bucket_end_utc": row.bucket_end_utc,
        "source_available_at_utc": row.joint_available_at_utc,
        "entry_time_utc": entry,
        "exit_time_utc": entry + HOLD,
        "side": side,
        "rank_L": int(row.q_rank_L if use_q_rank else row.rank_L),
        "rank_E": int(row.q_rank_E if use_q_rank else row.rank_E),
        "rank_n": int(row.q_rank_n if use_q_rank else row.rank_n),
        "rank": float(row.q_rank if use_q_rank else row.rank),
    }


def raw_candidates(features: pd.DataFrame, control: str) -> pd.DataFrame:
    if control not in INDEPENDENT_CONTROLS:
        raise ValueError("BFWC-288 raw candidates require independent control")
    rows: list[dict[str, Any]] = []
    records = list(features.itertuples(index=False))
    for index, row in enumerate(records):
        if not bool(row.base_valid) or row.rank_n < 120:
            continue
        rotation_ok = math.isfinite(row.rank) and row.rank >= 0.75 and row.R != 0.0
        witness_fullness_ok = (
            row.W != 0.0 and row.U != 0.0 and _side(row.W) == _side(row.U)
        )
        accepted = False
        side: str | None = None
        use_q = False
        if control == "primary":
            accepted = (
                rotation_ok
                and witness_fullness_ok
                and _side(row.R) == _side(row.W)
            )
            side = _side(row.R) if accepted else None
        elif control == "fee_rotation_only":
            accepted = rotation_ok
            side = _side(row.R) if accepted else None
        elif control == "witness_fullness_only":
            accepted = (
                row.q_rank_n >= 120
                and math.isfinite(row.q_rank)
                and row.q_rank >= 0.75
                and witness_fullness_ok
            )
            side = _side(row.W) if accepted else None
            use_q = True
        elif control == "drop_witness":
            accepted = (
                rotation_ok
                and row.U != 0.0
                and _side(row.R) == _side(row.U)
            )
            side = _side(row.R) if accepted else None
        elif control == "drop_fullness":
            accepted = (
                rotation_ok
                and row.W != 0.0
                and _side(row.R) == _side(row.W)
            )
            side = _side(row.R) if accepted else None
        elif control == "one_bucket_stale_witness_fullness":
            if index:
                stale = records[index - 1]
                accepted = (
                    rotation_ok
                    and bool(stale.base_valid)
                    and stale.W != 0.0
                    and stale.U != 0.0
                    and _side(row.R) == _side(stale.W) == _side(stale.U)
                )
                side = _side(row.R) if accepted else None
        if accepted and side is not None:
            rows.append(_raw_candidate(row, control, side, use_q_rank=use_q))
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def _assign_window(rows: pd.DataFrame) -> pd.DataFrame:
    assigned: list[pd.Series] = []
    for _, row in rows.iterrows():
        for name, (start, end) in SPLITS.items():
            if row["entry_time_utc"] >= start and row["exit_time_utc"] <= end:
                copied = row.copy()
                copied["window"] = name
                assigned.append(copied)
                break
    if not assigned:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    return pd.DataFrame(assigned, columns=CLOCK_COLUMNS)


def reserve_nonoverlap(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply split containment, then one global chronological reservation."""

    contained = _assign_window(rows)
    if contained.empty:
        return contained
    ordered = contained.sort_values(
        ["entry_time_utc", "bucket_start_utc", "signal_id"],
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
    result = primary.copy()
    result["control"] = control
    if control == "exact_direction_flip":
        result["side"] = result["side"].map({"LONG": "SHORT", "SHORT": "LONG"})
    elif control == "deterministic_random_side":
        result["side"] = result["signal_id"].map(prereg.deterministic_random_side)
    elif control == "constant_long":
        result["side"] = "LONG"
    elif control == "constant_short":
        result["side"] = "SHORT"
    elif control == "one_bar_delayed_entry":
        result["entry_time_utc"] = result["entry_time_utc"] + BAR
        result["exit_time_utc"] = result["exit_time_utc"] + BAR
        result = _assign_window(result)
    else:
        raise ValueError(f"BFWC-288 unknown parent control: {control}")
    return result.loc[:, CLOCK_COLUMNS].reset_index(drop=True)


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
        controls[control] = _parent_control(controls["primary"], control)
        raw_counts[control] = len(controls["primary"])
    return controls, raw_counts


def _clock_stats(rows: pd.DataFrame) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {
            "total": 0,
            "LONG": 0,
            "SHORT": 0,
            "maximum_month_share": None,
            "monthly_counts": {},
        }
    months = rows["entry_time_utc"].dt.strftime("%Y-%m").value_counts().sort_index()
    return {
        "total": count,
        "LONG": int(rows["side"].eq("LONG").sum()),
        "SHORT": int(rows["side"].eq("SHORT").sum()),
        "maximum_month_share": float(months.max() / count),
        "monthly_counts": {str(key): int(value) for key, value in months.items()},
    }


def support_checks(
    primary: pd.DataFrame,
    *,
    exact_join_gaps: int = 0,
    append_invariance_passed: bool = True,
) -> tuple[dict[str, Any], dict[str, bool]]:
    stats = {
        name: _clock_stats(primary.loc[primary["window"].eq(name)])
        for name in SPLITS
    }
    selection = primary.loc[primary["window"].eq("selection")]
    future_2025 = primary.loc[primary["window"].eq("future_2025")]
    future_2026 = primary.loc[primary["window"].eq("future_2026")]

    def count_between(rows: pd.DataFrame, start: str, end: str) -> int:
        entries = rows["entry_time_utc"]
        return int(
            (
                entries.ge(pd.Timestamp(start))
                & entries.lt(pd.Timestamp(end))
            ).sum()
        )

    checks = {
        "exact_join_gaps_zero": exact_join_gaps == 0,
        "future_append_invariance": bool(append_invariance_passed),
        "selection_total_min": stats["selection"]["total"] >= 45,
        "selection_2023_NovDec_min": count_between(
            selection, "2023-11-01T00:00:00Z", "2024-01-01T00:00:00Z"
        )
        >= 6,
        "selection_2024_H1_min": count_between(
            selection, "2024-01-01T00:00:00Z", "2024-07-01T00:00:00Z"
        )
        >= 12,
        "selection_2024_H2_min": count_between(
            selection, "2024-07-01T00:00:00Z", "2025-01-01T00:00:00Z"
        )
        >= 12,
        "selection_each_side_min": min(
            stats["selection"]["LONG"], stats["selection"]["SHORT"]
        )
        >= 14,
        "selection_maximum_month_share": bool(
            stats["selection"]["maximum_month_share"] is not None
            and stats["selection"]["maximum_month_share"] <= 0.20
        ),
        "future_2025_total_min": stats["future_2025"]["total"] >= 30,
        "future_2025_each_half_min": min(
            count_between(
                future_2025, "2025-01-01T00:00:00Z", "2025-07-01T00:00:00Z"
            ),
            count_between(
                future_2025, "2025-07-01T00:00:00Z", "2026-01-01T00:00:00Z"
            ),
        )
        >= 10,
        "future_2025_each_side_min": min(
            stats["future_2025"]["LONG"], stats["future_2025"]["SHORT"]
        )
        >= 10,
        "future_2025_maximum_month_share": bool(
            stats["future_2025"]["maximum_month_share"] is not None
            and stats["future_2025"]["maximum_month_share"] <= 0.25
        ),
        "future_2026_total_min": stats["future_2026"]["total"] >= 15,
        "future_2026_Q1_min": count_between(
            future_2026, "2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z"
        )
        >= 6,
        "future_2026_AprMay_min": count_between(
            future_2026, "2026-04-01T00:00:00Z", "2026-06-01T00:00:00Z"
        )
        >= 4,
        "future_2026_each_side_min": min(
            stats["future_2026"]["LONG"], stats["future_2026"]["SHORT"]
        )
        >= 5,
        "future_2026_maximum_month_share": bool(
            stats["future_2026"]["maximum_month_share"] is not None
            and stats["future_2026"]["maximum_month_share"] <= 0.30
        ),
    }
    return stats, checks


def _frame_hash(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    for column in selected.columns:
        if pd.api.types.is_datetime64_any_dtype(selected[column]):
            selected[column] = selected[column].map(_format_time)
    data = selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _prefix_snapshot(
    features: pd.DataFrame,
    controls: Mapping[str, pd.DataFrame],
    end: pd.Timestamp,
) -> dict[str, Any]:
    prior_features = features.loc[
        features["joint_available_at_utc"].lt(end)
    ]
    primary_raw = raw_candidates(prior_features, "primary")
    primary = controls["primary"].loc[
        controls["primary"]["source_available_at_utc"].lt(end)
    ]
    feature_fields = (
        "bucket_start_utc",
        "base_valid",
        "rank_L",
        "rank_E",
        "rank_n",
        "rank",
        "q_rank_L",
        "q_rank_E",
        "q_rank_n",
        "q_rank",
    )
    raw_fields = (
        "signal_id",
        "side",
        "entry_time_utc",
        "exit_time_utc",
        "rank_L",
        "rank_E",
        "rank_n",
    )
    accepted_fields = (
        "signal_id",
        "side",
        "entry_time_utc",
        "exit_time_utc",
    )
    return {
        "feature_rows": len(prior_features),
        "feature_rank_state_sha256": _frame_hash(prior_features, feature_fields),
        "raw_candidates": len(primary_raw),
        "raw_candidates_sha256": _frame_hash(primary_raw, raw_fields),
        "accepted_primary_rows": len(primary),
        "accepted_primary_sha256": _frame_hash(primary, accepted_fields),
    }


def future_append_invariance(
    bfrt: pd.DataFrame,
    wctr: pd.DataFrame,
) -> tuple[bool, dict[str, Any]]:
    full_features, _ = build_joint_features(bfrt, wctr, exact_domain=False)
    full_controls, _ = build_controls(full_features)
    report: dict[str, Any] = {}
    passed = True
    for end in (
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:00:00Z"),
    ):
        left = bfrt.loc[bfrt["available_at_utc"].lt(end)].reset_index(drop=True)
        right = wctr.loc[wctr["available_at_utc"].lt(end)].reset_index(drop=True)
        prefix_features, _ = build_joint_features(left, right, exact_domain=False)
        prefix_controls, _ = build_controls(prefix_features)
        expected = _prefix_snapshot(full_features, full_controls, end)
        rebuilt = _prefix_snapshot(prefix_features, prefix_controls, end)
        equal = expected == rebuilt
        passed = passed and equal
        report[_format_time(end)] = {
            "passed": equal,
            "completed_prefix": rebuilt,
            "full_rebuild_prefix": expected,
        }
    return passed, report


def _validate_clock_intervals(
    rows: pd.DataFrame,
    *,
    start: pd.Timestamp = FULL_START,
    end: pd.Timestamp = FULL_END,
) -> pd.DataFrame:
    required = ("entry_time_utc", "exit_time_utc", "side")
    if any(column not in rows.columns for column in required):
        raise RuntimeError("BFWC-288 comparator schema drift")
    validated = rows.loc[:, required].copy()
    for column in ("entry_time_utc", "exit_time_utc"):
        validated[column] = _parse_time_column(validated[column], column)
    if validated.empty:
        raise RuntimeError("BFWC-288 comparator is empty")
    if validated["side"].isna().any() or not validated["side"].isin(
        ("LONG", "SHORT")
    ).all():
        raise RuntimeError("BFWC-288 comparator side drift")
    validated = validated.sort_values("entry_time_utc", kind="mergesort").reset_index(
        drop=True
    )
    if validated["entry_time_utc"].duplicated().any():
        raise RuntimeError("BFWC-288 comparator duplicate entry")
    if not validated["exit_time_utc"].gt(validated["entry_time_utc"]).all():
        raise RuntimeError("BFWC-288 comparator invalid interval")
    epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    for column in ("entry_time_utc", "exit_time_utc"):
        if not (
            (validated[column] - epoch).map(lambda value: value % BAR)
            == pd.Timedelta(0)
        ).all():
            raise RuntimeError("BFWC-288 comparator off five-minute grid")
    if (
        validated["entry_time_utc"].lt(start).any()
        or validated["exit_time_utc"].gt(end).any()
    ):
        raise RuntimeError("BFWC-288 comparator outside full calendar")
    if len(validated) > 1 and not validated["entry_time_utc"].iloc[1:].reset_index(
        drop=True
    ).ge(validated["exit_time_utc"].iloc[:-1].reset_index(drop=True)).all():
        raise RuntimeError("BFWC-288 comparator overlap")
    return validated


def exact_entry_jaccard(left: pd.DataFrame, right: pd.DataFrame) -> float:
    first = set(left["entry_time_utc"])
    second = set(right["entry_time_utc"])
    union = first | second
    if not union:
        raise RuntimeError("BFWC-288 exact-entry Jaccard undefined")
    return float(len(first & second) / len(union))


def candidate_six_hour_containment(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
) -> float:
    if candidate.empty:
        raise RuntimeError("BFWC-288 containment denominator is zero")
    right = comparator["entry_time_utc"].sort_values().to_numpy()
    tolerance = np.timedelta64(6, "h")
    matches = 0
    for value in candidate["entry_time_utc"].to_numpy():
        index = int(np.searchsorted(right, value))
        neighbors = right[max(0, index - 1) : min(len(right), index + 1)]
        if any(abs(other - value) <= tolerance for other in neighbors):
            matches += 1
    return float(matches / len(candidate))


def _signed_occupancy(
    rows: pd.DataFrame,
    start: pd.Timestamp = FULL_START,
    end: pd.Timestamp = FULL_END,
) -> np.ndarray:
    validated = _validate_clock_intervals(rows, start=start, end=end)
    occupancy = np.zeros(int((end - start) / BAR), dtype=np.int8)
    for row in validated.itertuples(index=False):
        left = int((row.entry_time_utc - start) / BAR)
        right = int((row.exit_time_utc - start) / BAR)
        occupancy[left:right] = 1 if row.side == "LONG" else -1
    return occupancy


def absolute_signed_exposure_pearson(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    start: pd.Timestamp = FULL_START,
    end: pd.Timestamp = FULL_END,
) -> float:
    first = _signed_occupancy(left, start, end)
    second = _signed_occupancy(right, start, end)
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        raise RuntimeError("BFWC-288 signed exposure Pearson undefined")
    value = float(np.corrcoef(first, second)[0, 1])
    if not math.isfinite(value):
        raise RuntimeError("BFWC-288 signed exposure Pearson nonfinite")
    return abs(value)


def novelty_metrics(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
) -> dict[str, float]:
    left = _validate_clock_intervals(candidate)
    right = _validate_clock_intervals(comparator)
    return {
        "exact_entry_jaccard": exact_entry_jaccard(left, right),
        "candidate_6h_containment": candidate_six_hour_containment(left, right),
        "absolute_signed_exposure_pearson": absolute_signed_exposure_pearson(
            left, right
        ),
    }


def _prepare_comparator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    expected = {
        "clock",
        "entry_time_utc",
        "exit_time_utc",
        "side",
    }
    if set(frame.columns) != expected:
        raise RuntimeError("BFWC-288 comparator read schema drift")
    prepared = frame.copy()
    if not prepared["clock"].eq("primary").all():
        raise RuntimeError("BFWC-288 comparator clock drift")
    prepared["entry_time_utc"] = _parse_time_column(
        prepared["entry_time_utc"], "entry_time_utc"
    )
    prepared["exit_time_utc"] = _parse_time_column(
        prepared["exit_time_utc"], "exit_time_utc"
    )
    prepared["side"] = prepared["side"].map(
        {
            "1": "LONG",
            "-1": "SHORT",
            "LONG": "LONG",
            "SHORT": "SHORT",
        }
    )
    prepared = prepared.loc[
        prepared["entry_time_utc"].ge(FULL_START)
        & prepared["exit_time_utc"].le(FULL_END)
    ]
    return _validate_clock_intervals(
        prepared.loc[:, ["entry_time_utc", "exit_time_utc", "side"]]
    )


def _read_comparators(
    payload: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], int]:
    frames: dict[str, pd.DataFrame] = {}
    audit: dict[str, Any] = {}
    rows_loaded = 0
    for contract in payload["novelty"]["prior_primary_clocks"]:
        artifact = contract["artifact"]
        path = artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"BFWC-288 comparator hash drift: {contract['id']}")
        expected_header = ",".join(artifact["header"]) + "\n"
        _validate_exact_header(path, expected_header)
        frame = pd.read_csv(
            _path(path),
            usecols=[
                "clock",
                "entry_time_utc",
                "exit_time_utc",
                "side",
            ],
            dtype="string",
        )
        try:
            selected = _prepare_comparator_frame(frame)
        except RuntimeError as error:
            raise RuntimeError(
                f"BFWC-288 comparator validation failed: {contract['id']}"
            ) from error
        frames[contract["id"]] = selected
        rows_loaded += len(frame)
        audit[contract["id"]] = {
            "path": path,
            "sha256": artifact["sha256"],
            "header_sha256": artifact["header_sha256"],
            "rows_loaded": len(frame),
            "frame_sha256": _frame_hash(
                selected, ("entry_time_utc", "exit_time_utc", "side")
            ),
        }
    return frames, audit, rows_loaded


def evaluate_novelty(
    primary: pd.DataFrame,
    payload: Mapping[str, Any],
    *,
    comparator_loader: Callable[
        [Mapping[str, Any]],
        tuple[dict[str, pd.DataFrame], dict[str, Any], int],
    ] = _read_comparators,
) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any], int]:
    comparators, audit, rows_loaded = comparator_loader(payload)
    report: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    contracts = {
        row["id"]: row for row in payload["novelty"]["prior_primary_clocks"]
    }
    for comparator_id, comparator in comparators.items():
        metrics = novelty_metrics(primary, comparator)
        thresholds = contracts[comparator_id]["thresholds"]
        report[comparator_id] = {**metrics, "thresholds": thresholds}
        for metric, threshold in thresholds.items():
            checks[f"{comparator_id}:{metric}"] = metrics[metric] <= threshold
    return report, checks, audit, rows_loaded


def deterministic_clock_bytes(rows: pd.DataFrame) -> bytes:
    if list(rows.columns) != list(CLOCK_COLUMNS):
        raise RuntimeError("BFWC-288 clock schema drift")
    _reject_forbidden_columns(rows.columns)
    serialized = rows.copy()
    for column in (
        "bucket_start_utc",
        "bucket_end_utc",
        "source_available_at_utc",
        "entry_time_utc",
        "exit_time_utc",
    ):
        serialized[column] = serialized[column].map(_format_time)
    text = serialized.to_csv(
        index=False, columns=CLOCK_COLUMNS, lineterminator="\n"
    ).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer, mode="wb", filename="", mtime=0
    ) as zipped:
        zipped.write(text)
    return buffer.getvalue()


def _combined_controls(
    controls: Mapping[str, pd.DataFrame],
    names: Iterable[str] = CONTROL_ORDER,
) -> pd.DataFrame:
    frames = [
        controls[name] for name in names if not controls[name].empty
    ]
    if not frames:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["control", "entry_time_utc", "signal_id"], kind="mergesort"
    ).reset_index(drop=True)


def build_support_from_frames(
    bfrt: pd.DataFrame,
    wctr: pd.DataFrame,
    *,
    exact_domain: bool = False,
    artifact_eligible: bool = False,
    source_binding_audit: Mapping[str, Any] | None = None,
    comparator_loader: Callable[
        [Mapping[str, Any]],
        tuple[dict[str, pd.DataFrame], dict[str, Any], int],
    ] = _read_comparators,
) -> tuple[dict[str, Any], bytes, bytes]:
    payload = validate_preregistration()
    features, join_audit = build_joint_features(
        bfrt, wctr, exact_domain=exact_domain
    )
    controls, raw_counts = build_controls(features)
    append_passed, append_report = future_append_invariance(bfrt, wctr)
    statistics, checks = support_checks(
        controls["primary"],
        exact_join_gaps=join_audit["exact_join_gaps"],
        append_invariance_passed=append_passed,
    )
    support_passed = bool(checks and all(checks.values()))
    novelty_report: dict[str, Any] = {}
    novelty_checks: dict[str, bool] = {}
    comparator_audit: dict[str, Any] = {}
    comparator_rows = 0
    novelty_status = "not_opened"
    if support_passed and artifact_eligible:
        (
            novelty_report,
            novelty_checks,
            comparator_audit,
            comparator_rows,
        ) = evaluate_novelty(
            controls["primary"], payload, comparator_loader=comparator_loader
        )
        novelty_status = "passed" if all(novelty_checks.values()) else "failed"
    elif support_passed:
        novelty_status = "not_opened_synthetic"
    novelty_passed = bool(
        support_passed
        and artifact_eligible
        and novelty_checks
        and all(novelty_checks.values())
    )
    if not support_passed:
        decision = "retire_BFWC_288_unchanged"
        next_action = None
    elif not artifact_eligible:
        decision = "synthetic_validation_only"
        next_action = None
    elif not novelty_passed:
        decision = "retire_BFWC_288_unchanged"
        next_action = None
    else:
        decision = "advance_to_separately_committed_Gross9_novelty_evaluator"
        next_action = "freeze_Gross9_novelty_evaluator"
    primary_bytes = deterministic_clock_bytes(controls["primary"])
    control_bytes = deterministic_clock_bytes(
        _combined_controls(
            controls,
            (name for name in CONTROL_ORDER if name != "primary"),
        )
    )
    source_rows = {"BFRT": len(bfrt), "WCTR": len(wctr)}
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "artifact_eligible": artifact_eligible,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "implementation": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
            "tests": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in TEST_PATHS
            ],
        },
        "rows_loaded": {
            "BFRT_source": source_rows["BFRT"],
            "WCTR_source": source_rows["WCTR"],
            "BFRT_comparator": comparator_audit.get("BFRT-288", {}).get(
                "rows_loaded", 0
            ),
            "WCTR_comparator": comparator_audit.get("WCTR-288", {}).get(
                "rows_loaded", 0
            ),
            "comparator_total": comparator_rows,
            "market": 0,
            "funding": 0,
            "premium": 0,
            "returns": 0,
            "Gross9": 0,
        },
        "source_features_computed": True,
        "source_incidence_opened": True,
        "candidate_overlap_opened": support_passed and artifact_eligible,
        "economic_rows_opened": False,
        "outcomes_opened": False,
        "join_audit": join_audit,
        "source_binding_audit": dict(source_binding_audit or {}),
        "frame_hashes": {
            "BFRT_source_usecols": _frame_hash(bfrt, BFRT_USECOLS),
            "WCTR_source_usecols": _frame_hash(wctr, WCTR_USECOLS),
            "features": _frame_hash(features, FEATURE_COLUMNS),
        },
        "feature_counts": {
            "joined": len(features),
            "base_valid": int(features["base_valid"].sum()),
            "raw_candidates": raw_counts,
            "accepted_clocks": {
                name: len(controls[name]) for name in CONTROL_ORDER
            },
        },
        "support_statistics": statistics,
        "support_checks": checks,
        "future_append_invariance": append_report,
        "support_passed": support_passed,
        "novelty_status": novelty_status,
        "comparator_audit": comparator_audit,
        "novelty_report": novelty_report,
        "novelty_checks": novelty_checks,
        "novelty_passed": novelty_passed,
        "clock_artifacts": {
            "primary": {
                "path": str(DEFAULT_PRIMARY_CLOCK_OUTPUT),
                "sha256": hashlib.sha256(primary_bytes).hexdigest(),
                "rows": len(controls["primary"]),
                "columns": list(CLOCK_COLUMNS),
            },
            "controls": {
                "path": str(DEFAULT_CONTROL_CLOCK_OUTPUT),
                "sha256": hashlib.sha256(control_bytes).hexdigest(),
                "rows": sum(
                    len(controls[name])
                    for name in CONTROL_ORDER
                    if name != "primary"
                ),
                "columns": list(CLOCK_COLUMNS),
            },
        },
        "decision": decision,
        "next_action": next_action,
        "outcome_boundary": {
            "source_rows_loaded": source_rows,
            "comparator_rows_loaded_conditionally": comparator_rows,
            "market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "premium_rows_loaded": 0,
            "return_rows_loaded": 0,
            "outcome_columns_loaded": 0,
            "outcomes_computed": False,
            "network_calls": 0,
        },
    }
    return (
        {**core, "manifest_hash": canonical_hash(core)},
        primary_bytes,
        control_bytes,
    )


def build_real_support_payload() -> tuple[dict[str, Any], bytes, bytes]:
    _assert_protocol_committed()
    payload = validate_preregistration()
    source_binding_audit = validate_source_bindings()
    bfrt, wctr = load_sources()
    report, primary, controls = build_support_from_frames(
        bfrt,
        wctr,
        exact_domain=True,
        artifact_eligible=True,
        source_binding_audit=source_binding_audit,
    )
    if report["preregistration"]["manifest_hash"] != payload["manifest_hash"]:
        raise RuntimeError("BFWC-288 preregistration binding changed during build")
    return report, primary, controls


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


def write_once(path: str | Path, data: bytes) -> str:
    output = _path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != data:
            raise RuntimeError(f"BFWC-288 noncanonical existing artifact: {path}")
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
                raise RuntimeError(f"BFWC-288 artifact race drift: {path}")
            return "verified_existing"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


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
        raise RuntimeError("BFWC-288 result artifact paths are frozen")
    report, primary_bytes, control_bytes = build_real_support_payload()
    primary_status = write_once(primary_output, primary_bytes)
    controls_status = write_once(controls_output, control_bytes)
    report_status = write_once(report_output, _json_bytes(report))
    return {
        "report_status": report_status,
        "primary_clock_status": primary_status,
        "control_clocks_status": controls_status,
        "support_passed": report["support_passed"],
        "novelty_passed": report["novelty_passed"],
        "decision": report["decision"],
        "manifest_hash": report["manifest_hash"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(write_support(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
