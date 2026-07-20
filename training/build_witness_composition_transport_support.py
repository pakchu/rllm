"""Build outcome-blind WCTR-288 feature and support clocks.

This stage opens only the hash-bound mined-block size/weight source. It derives
frozen source features, primary incidence, and diagnostic controls; it never
loads BTC market data, funding, returns, or PnL.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import preregister_witness_composition_transport as prereg


POLICY_ID = "WCTR-288"
PROTOCOL_VERSION = "witness_composition_transport_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = Path(
    "results/witness_composition_transport_preregistration_2026-07-20.json"
)
DEFAULT_OUTPUT = Path(
    "results/witness_composition_transport_support_2026-07-20.json"
)
DEFAULT_PRIMARY_CLOCK = Path(
    "results/witness_composition_transport_primary_clock_2026-07-20.csv.gz"
)
DEFAULT_CONTROL_CLOCKS = Path(
    "results/witness_composition_transport_control_clocks_2026-07-20.csv.gz"
)
SUPPORT_BUILDER = Path(
    "training/build_witness_composition_transport_support.py"
)
EXPECTED_PREREGISTRATION_FILE_SHA256 = (
    "f1d8d5641f1773d00dc2a99a4ca7e11b68cbc5b0cebc1456514fcbbcd9c9d3d1"
)
EXPECTED_PREREGISTRATION_MANIFEST_HASH = (
    "1c844325ea3134683cd89a071e007cec70b475b641376c111141a524f2a07a78"
)
EXPECTED_POLICY_HASH = (
    "510cedafde2775d65e3bc77eaefeccb9d526b9d738e503aa7c6c0e277974ddeb"
)
EXPECTED_PREREGISTRATION_SOURCE_SHA256 = (
    "58a3725d8e64b47181a8e9310c370c583084fa27ea9c633f6e6c41dae266bf20"
)
SOURCE_COLUMNS = list(prereg.SOURCE_COLUMNS)
INTEGER_COLUMNS = ["avg_height", "avg_timestamp", "avg_size", "avg_weight"]
FEATURE_COLUMNS = [
    "witness_share",
    "fullness",
    "transport_7d",
    "impulse_24h",
    "log_size_7d",
    "log_size_24h",
    "log_weight_7d",
    "log_weight_24h",
    "magnitude_rank",
    "fullness_rank",
    "impulse_magnitude_rank",
    "log_size_magnitude_rank",
    "log_weight_magnitude_rank",
]
CLOCK_COLUMNS = [
    "policy_id",
    "clock",
    "window",
    "bucket_start_utc",
    "source_available_at_utc",
    "entry_time_utc",
    "exit_time_utc",
    "side",
    *FEATURE_COLUMNS,
]
CONTROL_NAMES = [
    "direction_flip",
    "transport_only",
    "impulse_only",
    "low_fullness_complement",
    "serialized_size_only",
    "block_weight_only",
    "constant_long_same_clock",
    "constant_short_same_clock",
    "stale_7d",
    "month_side_stratified_random_clock",
    "one_bar_delayed_entry",
]
OUTCOME_BOUNDARY = {
    "source_values_read": True,
    "source_feature_rows_derived": True,
    "signal_incidence_rows_derived": True,
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "premium_or_oi_rows_loaded": 0,
    "return_rows_loaded": 0,
    "market_values_read": 0,
    "funding_values_read": 0,
    "return_or_pnl_fields": 0,
}
FORBIDDEN_SOURCE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "price",
    "return",
    "returns",
    "pnl",
    "funding",
    "funding_rate",
    "premium",
    "open_interest",
    "oi",
}
WINDOWS = {
    "train": (
        pd.Timestamp("2022-11-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:00:00Z"),
    ),
    "test": (
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
    ),
    "eval": (
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:00:00Z"),
    ),
    "forward": (
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-07-20T00:00:00Z"),
    ),
}


@dataclass(frozen=True)
class Config:
    preregistration: str = str(DEFAULT_PREREGISTRATION)
    output: str = str(DEFAULT_OUTPUT)
    primary_clock: str = str(DEFAULT_PRIMARY_CLOCK)
    control_clocks: str = str(DEFAULT_CONTROL_CLOCKS)
    artifact_root: str = "results"


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"WCTR JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _expected_source_binding() -> dict[str, Any]:
    return {
        "path": str(prereg.SOURCE_MANIFEST),
        "sha256": prereg.EXPECTED_SOURCE_MANIFEST_FILE_SHA256,
        "manifest_hash": prereg.EXPECTED_SOURCE_MANIFEST_HASH,
        "protocol_version": prereg.SOURCE_PROTOCOL_VERSION,
        "source_decision": prereg.EXPECTED_SOURCE_DECISION,
        "source_freeze": {
            "path": str(prereg.SOURCE_FREEZE),
            "sha256": prereg.SOURCE_FREEZE_SHA256,
        },
        "source_builder": prereg.EXPECTED_SOURCE_BUILDER,
        "source_audit": prereg.EXPECTED_SOURCE_AUDIT,
        "raw_artifact": prereg.EXPECTED_RAW_ARTIFACT,
        "normalized_artifact": prereg.EXPECTED_NORMALIZED_ARTIFACT,
        "source_semantics": prereg.EXPECTED_SOURCE_SEMANTICS,
        "causal_availability": prereg.EXPECTED_CAUSAL_AVAILABILITY,
        "outcome_boundary": prereg.SOURCE_OUTCOME_BOUNDARY,
    }


def validate_frozen_preregistration(path: str | Path) -> dict[str, Any]:
    artifact_path = _repository_path(path)
    if artifact_path != _repository_path(DEFAULT_PREREGISTRATION):
        raise RuntimeError("WCTR preregistration path differs from frozen artifact")
    artifact_bytes = artifact_path.read_bytes()
    if (
        hashlib.sha256(artifact_bytes).hexdigest()
        != EXPECTED_PREREGISTRATION_FILE_SHA256
    ):
        raise RuntimeError("WCTR preregistration file SHA drift")
    artifact = json.loads(
        artifact_bytes.decode("utf-8"),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(artifact, dict):
        raise RuntimeError("WCTR preregistration must be a JSON object")
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    if canonical_hash(core) != artifact.get("manifest_hash"):
        raise RuntimeError("WCTR preregistration canonical hash mismatch")
    if artifact.get("protocol_version") != prereg.PROTOCOL_VERSION:
        raise RuntimeError("WCTR preregistration protocol drift")
    if artifact.get("manifest_hash") != EXPECTED_PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("WCTR preregistration manifest hash drift")
    if artifact.get("policy_hash") != EXPECTED_POLICY_HASH:
        raise RuntimeError("WCTR policy hash drift")
    if canonical_hash(artifact.get("policy")) != EXPECTED_POLICY_HASH:
        raise RuntimeError("WCTR policy content drift")
    expected_source = {
        "path": str(prereg.PREREGISTRATION_SOURCE),
        "sha256": EXPECTED_PREREGISTRATION_SOURCE_SHA256,
    }
    if artifact.get("preregistration_source") != expected_source:
        raise RuntimeError("WCTR preregistration source binding drift")
    if (
        sha256_file(prereg.PREREGISTRATION_SOURCE)
        != EXPECTED_PREREGISTRATION_SOURCE_SHA256
    ):
        raise RuntimeError("WCTR preregistration source file drift")
    if artifact.get("policy") != prereg.policy():
        raise RuntimeError("WCTR policy singleton drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("WCTR preregistration opened outcomes")
    if artifact.get("outcome_boundary") != prereg.PREREGISTRATION_OUTCOME_BOUNDARY:
        raise RuntimeError("WCTR preregistration outcome boundary drift")
    source = artifact.get("source_manifest")
    if not isinstance(source, dict):
        raise RuntimeError("WCTR source binding missing")
    if source != _expected_source_binding():
        raise RuntimeError("WCTR frozen source metadata binding drift")
    return artifact


def _source_binding(artifact: dict[str, Any]) -> dict[str, Any]:
    source = artifact["source_manifest"].get("normalized_artifact")
    if not isinstance(source, dict):
        raise RuntimeError("WCTR normalized source binding missing")
    return source


def _validate_config(cfg: Config, preregistration: dict[str, Any]) -> None:
    paths = {
        "output": _repository_path(cfg.output),
        "primary_clock": _repository_path(cfg.primary_clock),
        "control_clocks": _repository_path(cfg.control_clocks),
    }
    if not str(cfg.output).endswith(".json"):
        raise ValueError("WCTR support output must be JSON")
    for name in ("primary_clock", "control_clocks"):
        if not str(getattr(cfg, name)).endswith(".csv.gz"):
            raise ValueError(f"WCTR {name} must be .csv.gz")
    if len(set(paths.values())) != len(paths):
        raise ValueError("WCTR support output paths must be distinct")
    protected = {
        _repository_path(cfg.preregistration),
        _repository_path(prereg.SOURCE_MANIFEST),
        _repository_path(_source_binding(preregistration)["path"]),
        _repository_path(prereg.EXPECTED_RAW_ARTIFACT["path"]),
        _repository_path(prereg.SOURCE_DECISION),
        _repository_path(prereg.SOURCE_FREEZE),
        _repository_path(prereg.SOURCE_BUILDER),
        _repository_path(prereg.PREREGISTRATION_SOURCE),
        _repository_path(prereg.PREREGISTRATION_DOCUMENT),
        _repository_path(SUPPORT_BUILDER),
    }
    if set(paths.values()) & protected:
        raise ValueError("WCTR support output aliases a frozen input")
    artifact_root = _repository_path(cfg.artifact_root)
    outside = [
        name for name, path in paths.items() if not path.is_relative_to(artifact_root)
    ]
    if outside:
        raise ValueError(
            "WCTR support outputs must stay under the explicit artifact root; "
            f"outside={outside!r}"
        )
    existing = [name for name, path in paths.items() if path.exists()]
    if existing:
        raise FileExistsError(
            f"WCTR support artifacts are immutable; existing={existing!r}"
        )


def validate_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != SOURCE_COLUMNS:
        raise RuntimeError("WCTR source frame schema drift")
    if frame.empty:
        raise RuntimeError("WCTR source frame is empty")
    forbidden = FORBIDDEN_SOURCE_COLUMNS.intersection(
        {str(column).lower() for column in frame.columns}
    )
    if forbidden:
        raise RuntimeError(f"WCTR source contains outcome-like columns: {forbidden!r}")
    out = frame.copy()
    for column in INTEGER_COLUMNS:
        values = pd.to_numeric(out[column], errors="raise")
        if values.isna().any() or (values % 1).ne(0).any():
            raise RuntimeError(f"WCTR source {column} must contain exact integers")
        if (values <= 0).any():
            raise RuntimeError(f"WCTR source {column} must be positive")
        out[column] = values.astype(np.int64)

    for column in ("bucket_start_utc", "bucket_end_utc", "available_at_utc"):
        parsed = pd.to_datetime(out[column], utc=True, errors="raise")
        if parsed.isna().any():
            raise RuntimeError(f"WCTR source {column} contains null timestamps")
        out[column] = parsed
    if out["bucket_start_utc"].duplicated().any():
        raise RuntimeError("WCTR source bucket starts must be unique")
    if not out["bucket_start_utc"].is_monotonic_increasing:
        raise RuntimeError("WCTR source bucket starts must be strictly increasing")
    out = out.reset_index(drop=True)
    if (
        not out["avg_height"].is_monotonic_increasing
        or out["avg_height"].duplicated().any()
    ):
        raise RuntimeError("WCTR source average heights must be strictly increasing")
    if (
        not out["avg_timestamp"].is_monotonic_increasing
        or out["avg_timestamp"].duplicated().any()
    ):
        raise RuntimeError("WCTR source average timestamps must be strictly increasing")

    expected_end = out["bucket_start_utc"] + pd.Timedelta(hours=12)
    expected_available = expected_end + pd.Timedelta(hours=48)
    if not out["bucket_end_utc"].equals(expected_end):
        raise RuntimeError("WCTR source bucket-end clock drift")
    if not out["available_at_utc"].equals(expected_available):
        raise RuntimeError("WCTR source availability clock drift")
    unix_start = (
        out["bucket_start_utc"].astype("int64") // 1_000_000_000
    ).astype(np.int64)
    unix_end = (
        out["bucket_end_utc"].astype("int64") // 1_000_000_000
    ).astype(np.int64)
    timestamp_in_bucket = (out["avg_timestamp"] >= unix_start) & (
        out["avg_timestamp"] < unix_end
    )
    if not timestamp_in_bucket.all():
        raise RuntimeError("WCTR average timestamp escaped its derived bucket")
    deltas = out["bucket_start_utc"].diff().dropna()
    if not deltas.eq(pd.Timedelta(hours=12)).all():
        raise RuntimeError("WCTR source has a missing or irregular 12h bucket")

    size = out["avg_size"].to_numpy(np.int64)
    weight = out["avg_weight"].to_numpy(np.int64)
    if np.any(size > 4_000_000) or np.any(weight > 4_000_000):
        raise RuntimeError("WCTR source size/weight exceeds frozen bounds")
    if np.any(weight < size) or np.any(weight > 4 * size):
        raise RuntimeError("WCTR source violates BIP141 size/weight bounds")
    witness_share = (4.0 * size - weight) / (3.0 * size)
    fullness = weight / 4_000_000.0
    if (
        not np.isfinite(witness_share).all()
        or np.any(witness_share < 0.0)
        or np.any(witness_share > 1.0)
    ):
        raise RuntimeError("WCTR source implied witness share is invalid")
    if (
        not np.isfinite(fullness).all()
        or np.any(fullness <= 0.0)
        or np.any(fullness > 1.0)
    ):
        raise RuntimeError("WCTR source fullness is invalid")
    return out


def load_source_frame(preregistration: dict[str, Any]) -> pd.DataFrame:
    binding = _source_binding(preregistration)
    path = _repository_path(binding["path"])
    compressed = path.read_bytes()
    if len(compressed) != binding["bytes"]:
        raise RuntimeError("WCTR normalized source byte-size drift")
    if hashlib.sha256(compressed).hexdigest() != binding["sha256"]:
        raise RuntimeError("WCTR normalized source SHA drift")
    frame = pd.read_csv(io.BytesIO(compressed), compression="gzip")
    if frame.columns.tolist() != SOURCE_COLUMNS:
        raise RuntimeError("WCTR normalized source header drift")
    return validate_source_frame(frame)


def strict_prior_midrank(current: float, prior: Iterable[float]) -> float:
    values = list(prior)
    if not values:
        raise ValueError("WCTR strict-prior midrank requires prior values")
    less = sum(value < current for value in values)
    equal = sum(value == current for value in values)
    return (less + 0.5 * equal) / len(values)


def _witness_share(size: int, weight: int) -> float:
    return ((4.0 * size) - weight) / (3.0 * size)


def build_features(
    source: pd.DataFrame,
    *,
    lookback: int = 180,
    minimum_prior: int = 120,
) -> pd.DataFrame:
    frame = validate_source_frame(source)
    if lookback < minimum_prior or minimum_prior <= 0:
        raise ValueError("WCTR rank lookback/minimum contract is invalid")
    records: list[dict[str, Any]] = []
    histories: dict[str, list[float]] = {
        "magnitude_rank": [],
        "fullness_rank": [],
        "impulse_magnitude_rank": [],
        "log_size_magnitude_rank": [],
        "log_weight_magnitude_rank": [],
    }
    prior_available: list[pd.Timestamp] = []
    for index, row in frame.iterrows():
        record = row.to_dict()
        feature_values = {column: math.nan for column in FEATURE_COLUMNS}
        base_valid = False
        if index >= 14:
            starts = frame.loc[index - 14 : index, "bucket_start_utc"]
            contiguous = starts.diff().dropna().eq(pd.Timedelta(hours=12)).all()
            if contiguous:
                size_t = int(row["avg_size"])
                weight_t = int(row["avg_weight"])
                size_2 = int(frame.at[index - 2, "avg_size"])
                weight_2 = int(frame.at[index - 2, "avg_weight"])
                size_14 = int(frame.at[index - 14, "avg_size"])
                weight_14 = int(frame.at[index - 14, "avg_weight"])
                witness_t = _witness_share(size_t, weight_t)
                witness_2 = _witness_share(size_2, weight_2)
                witness_14 = _witness_share(size_14, weight_14)
                fullness = weight_t / 4_000_000.0
                transport = witness_t - witness_14
                impulse = witness_t - witness_2
                log_size_7d = math.log(size_t) - math.log(size_14)
                log_size_24h = math.log(size_t) - math.log(size_2)
                log_weight_7d = math.log(weight_t) - math.log(weight_14)
                log_weight_24h = math.log(weight_t) - math.log(weight_2)
                raw_features = {
                    "witness_share": witness_t,
                    "fullness": fullness,
                    "transport_7d": transport,
                    "impulse_24h": impulse,
                    "log_size_7d": log_size_7d,
                    "log_size_24h": log_size_24h,
                    "log_weight_7d": log_weight_7d,
                    "log_weight_24h": log_weight_24h,
                }
                base_valid = bool(
                    all(math.isfinite(value) for value in raw_features.values())
                    and 0.0 <= witness_t <= 1.0
                    and 0.0 <= witness_2 <= 1.0
                    and 0.0 <= witness_14 <= 1.0
                    and 0.0 <= fullness <= 1.0
                )
                if base_valid:
                    feature_values.update(raw_features)

        available = pd.Timestamp(record["available_at_utc"])
        rank_ready = False
        if base_valid and len(histories["magnitude_rank"]) >= minimum_prior:
            if prior_available and not prior_available[-1] < available:
                raise RuntimeError("WCTR strict-prior availability order drift")
            current_rank_values = {
                "magnitude_rank": abs(feature_values["transport_7d"]),
                "fullness_rank": feature_values["fullness"],
                "impulse_magnitude_rank": abs(feature_values["impulse_24h"]),
                "log_size_magnitude_rank": abs(feature_values["log_size_7d"]),
                "log_weight_magnitude_rank": abs(feature_values["log_weight_7d"]),
            }
            for rank_name, current in current_rank_values.items():
                feature_values[rank_name] = strict_prior_midrank(
                    current, histories[rank_name][-lookback:]
                )
            rank_ready = all(
                math.isfinite(feature_values[name]) for name in histories
            )
        if base_valid:
            histories["magnitude_rank"].append(abs(feature_values["transport_7d"]))
            histories["fullness_rank"].append(feature_values["fullness"])
            histories["impulse_magnitude_rank"].append(
                abs(feature_values["impulse_24h"])
            )
            histories["log_size_magnitude_rank"].append(
                abs(feature_values["log_size_7d"])
            )
            histories["log_weight_magnitude_rank"].append(
                abs(feature_values["log_weight_7d"])
            )
            prior_available.append(available)
        entry = available.ceil("5min") + pd.Timedelta(minutes=5)
        record.update(feature_values)
        record.update(
            {
                "entry_time_utc": entry,
                "exit_time_utc": entry + pd.Timedelta(hours=24),
                "feature_valid": base_valid,
                "rank_ready": rank_ready,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _window_name(entry: pd.Timestamp, exit_time: pd.Timestamp) -> str | None:
    for name, (start, end) in WINDOWS.items():
        if start <= entry and exit_time <= end:
            return name
    return None


def _same_nonzero_sign(left: float, right: float) -> bool:
    return bool(
        left != 0.0
        and right != 0.0
        and math.copysign(1.0, left) == math.copysign(1.0, right)
    )


def _candidate_side(row: Any, mode: str) -> int:
    if not bool(row.rank_ready):
        return 0
    transport = float(row.transport_7d)
    impulse = float(row.impulse_24h)
    fullness_rank = float(row.fullness_rank)
    if mode == "primary":
        eligible = bool(
            float(row.magnitude_rank) >= 0.75
            and fullness_rank >= 0.50
            and _same_nonzero_sign(transport, impulse)
        )
        side_value = transport
    elif mode == "transport_only":
        eligible = bool(transport != 0.0 and float(row.magnitude_rank) >= 0.75)
        side_value = transport
    elif mode == "impulse_only":
        eligible = bool(
            impulse != 0.0
            and float(row.impulse_magnitude_rank) >= 0.75
            and fullness_rank >= 0.50
        )
        side_value = impulse
    elif mode == "low_fullness_complement":
        eligible = bool(
            float(row.magnitude_rank) >= 0.75
            and fullness_rank < 0.50
            and _same_nonzero_sign(transport, impulse)
        )
        side_value = transport
    elif mode == "serialized_size_only":
        seven_day = float(row.log_size_7d)
        one_day = float(row.log_size_24h)
        eligible = bool(
            float(row.log_size_magnitude_rank) >= 0.75
            and fullness_rank >= 0.50
            and _same_nonzero_sign(seven_day, one_day)
        )
        side_value = seven_day
    elif mode == "block_weight_only":
        seven_day = float(row.log_weight_7d)
        one_day = float(row.log_weight_24h)
        eligible = bool(
            float(row.log_weight_magnitude_rank) >= 0.75
            and fullness_rank >= 0.50
            and _same_nonzero_sign(seven_day, one_day)
        )
        side_value = seven_day
    elif mode == "rank_ready":
        eligible = True
        side_value = 1.0
    else:
        raise ValueError(f"unknown WCTR candidate mode {mode!r}")
    if not eligible:
        return 0
    return 1 if side_value > 0.0 else -1


def _clock_record(row: Any, *, clock: str, window: str, side: int) -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "clock": clock,
        "window": window,
        "bucket_start_utc": row.bucket_start_utc,
        "source_available_at_utc": row.available_at_utc,
        "entry_time_utc": row.entry_time_utc,
        "exit_time_utc": row.exit_time_utc,
        "side": int(side),
        **{column: float(getattr(row, column)) for column in FEATURE_COLUMNS},
    }


def build_clock(features: pd.DataFrame, *, mode: str, clock: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    prior_exit: dict[str, pd.Timestamp] = {}
    ordered = features.sort_values(
        ["entry_time_utc", "bucket_start_utc"], kind="stable"
    )
    for row in ordered.itertuples(index=False):
        side = _candidate_side(row, mode)
        if side == 0:
            continue
        entry = pd.Timestamp(row.entry_time_utc)
        exit_time = pd.Timestamp(row.exit_time_utc)
        window = _window_name(entry, exit_time)
        if window is None:
            continue
        last_exit = prior_exit.get(window)
        if last_exit is not None and entry < last_exit:
            continue
        records.append(_clock_record(row, clock=clock, window=window, side=side))
        prior_exit[window] = exit_time
    return pd.DataFrame.from_records(records, columns=CLOCK_COLUMNS)


def _same_clock(primary: pd.DataFrame, name: str, sides: Iterable[int]) -> pd.DataFrame:
    control = primary.copy()
    control["clock"] = name
    control["side"] = list(sides)
    return control[CLOCK_COLUMNS]


def _stale_clock(features: pd.DataFrame) -> pd.DataFrame:
    stale = features.copy()
    shifted = features[["bucket_start_utc", *FEATURE_COLUMNS, "rank_ready"]].shift(14)
    exact_lag = (
        stale["bucket_start_utc"] - shifted["bucket_start_utc"]
    ).eq(pd.Timedelta(days=7))
    for column in [*FEATURE_COLUMNS, "rank_ready"]:
        stale[column] = shifted[column]
    stale["rank_ready"] = shifted["rank_ready"].eq(True) & exact_lag
    return build_clock(stale, mode="primary", clock="stale_7d")


def _random_key(window: str, month: str, entry: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(entry).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = f"20260720|{window}|{month}|{timestamp}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _random_clock(features: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    baseline = build_clock(features, mode="rank_ready", clock="random_pool")
    records: list[pd.DataFrame] = []
    if primary.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    primary = primary.copy()
    baseline = baseline.copy()
    primary["_month"] = primary["entry_time_utc"].dt.strftime("%Y-%m")
    baseline["_month"] = baseline["entry_time_utc"].dt.strftime("%Y-%m")
    for (window, month), target in primary.groupby(
        ["window", "_month"], sort=True, observed=True
    ):
        candidates = baseline[
            (baseline["window"] == window) & (baseline["_month"] == month)
        ].copy()
        if len(candidates) < len(target):
            raise RuntimeError("WCTR random control month pool is too small")
        candidates["_key"] = [
            _random_key(str(window), str(month), value)
            for value in candidates["entry_time_utc"]
        ]
        sampled = candidates.sort_values(
            ["_key", "entry_time_utc"], kind="stable"
        ).head(len(target)).copy()
        long_count = int((target["side"] == 1).sum())
        sampled["side"] = [-1] * len(sampled)
        sampled.iloc[:long_count, sampled.columns.get_loc("side")] = 1
        sampled["clock"] = "month_side_stratified_random_clock"
        records.append(sampled[CLOCK_COLUMNS])
    return (
        pd.concat(records, ignore_index=True)[CLOCK_COLUMNS]
        .sort_values(["entry_time_utc", "bucket_start_utc"], kind="stable")
        .reset_index(drop=True)
    )


def build_control_clocks(features: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    controls = [
        _same_clock(primary, "direction_flip", -primary["side"].astype(int)),
        build_clock(features, mode="transport_only", clock="transport_only"),
        build_clock(features, mode="impulse_only", clock="impulse_only"),
        build_clock(
            features,
            mode="low_fullness_complement",
            clock="low_fullness_complement",
        ),
        build_clock(
            features, mode="serialized_size_only", clock="serialized_size_only"
        ),
        build_clock(features, mode="block_weight_only", clock="block_weight_only"),
        _same_clock(primary, "constant_long_same_clock", [1] * len(primary)),
        _same_clock(primary, "constant_short_same_clock", [-1] * len(primary)),
        _stale_clock(features),
        _random_clock(features, primary),
    ]
    controls.append(_delayed_clock(primary))
    nonempty = [control for control in controls if not control.empty]
    combined = (
        pd.concat(nonempty, ignore_index=True)
        if nonempty
        else pd.DataFrame(columns=CLOCK_COLUMNS)
    )
    observed = set(combined["clock"].unique())
    if not observed.issubset(set(CONTROL_NAMES)):
        raise RuntimeError("WCTR control family drift")
    return combined.sort_values(
        ["clock", "entry_time_utc", "bucket_start_utc"], kind="stable"
    ).reset_index(drop=True)


def _delayed_clock(primary: pd.DataFrame) -> pd.DataFrame:
    delayed = primary.copy()
    delayed["clock"] = "one_bar_delayed_entry"
    delayed["entry_time_utc"] += pd.Timedelta(minutes=5)
    delayed["exit_time_utc"] += pd.Timedelta(minutes=5)
    contained = [
        _window_name(row.entry_time_utc, row.exit_time_utc) == row.window
        for row in delayed.itertuples(index=False)
    ]
    return delayed.loc[contained, CLOCK_COLUMNS].reset_index(drop=True)


def _subset(
    clock: pd.DataFrame, start: str, end: str
) -> pd.DataFrame:
    entry = clock["entry_time_utc"]
    return clock[
        (entry >= pd.Timestamp(start)) & (entry < pd.Timestamp(end))
    ]


def _max_month_share(clock: pd.DataFrame) -> float:
    if clock.empty:
        return 1.0
    return float(
        clock["entry_time_utc"].dt.strftime("%Y-%m").value_counts(normalize=True).max()
    )


def support_gate_summary(
    clock: pd.DataFrame,
    source: pd.DataFrame,
    *,
    control_name: str | None = None,
) -> dict[str, Any]:
    windows = {
        name: clock[clock["window"] == name]
        for name in ("train", "test", "eval", "forward")
    }
    periods = {
        "train_2022_nov_dec": _subset(
            clock, "2022-11-01T00:00:00Z", "2023-01-01T00:00:00Z"
        ),
        "train_2023_h1": _subset(
            clock, "2023-01-01T00:00:00Z", "2023-07-01T00:00:00Z"
        ),
        "train_2023_h2": _subset(
            clock, "2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"
        ),
        "test_2024_h1": _subset(
            clock, "2024-01-01T00:00:00Z", "2024-07-01T00:00:00Z"
        ),
        "test_2024_h2": _subset(
            clock, "2024-07-01T00:00:00Z", "2025-01-01T00:00:00Z"
        ),
        "eval_2025_h1": _subset(
            clock, "2025-01-01T00:00:00Z", "2025-07-01T00:00:00Z"
        ),
        "eval_2025_h2": _subset(
            clock, "2025-07-01T00:00:00Z", "2026-01-01T00:00:00Z"
        ),
        "forward_2026_h1": _subset(
            clock, "2026-01-01T00:00:00Z", "2026-07-01T00:00:00Z"
        ),
    }
    quarters = {
        quarter: _subset(clock, start, end)
        for quarter, start, end in (
            ("2024Q1", "2024-01-01T00:00:00Z", "2024-04-01T00:00:00Z"),
            ("2024Q2", "2024-04-01T00:00:00Z", "2024-07-01T00:00:00Z"),
            ("2024Q3", "2024-07-01T00:00:00Z", "2024-10-01T00:00:00Z"),
            ("2024Q4", "2024-10-01T00:00:00Z", "2025-01-01T00:00:00Z"),
            ("2025Q1", "2025-01-01T00:00:00Z", "2025-04-01T00:00:00Z"),
            ("2025Q2", "2025-04-01T00:00:00Z", "2025-07-01T00:00:00Z"),
            ("2025Q3", "2025-07-01T00:00:00Z", "2025-10-01T00:00:00Z"),
            ("2025Q4", "2025-10-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
    }
    side_counts = {
        name: {
            "long": int((window["side"] == 1).sum()),
            "short": int((window["side"] == -1).sum()),
        }
        for name, window in windows.items()
    }
    checks = {
        "train_total_minimum": len(windows["train"]) >= 45,
        "train_2022_nov_dec_minimum": len(periods["train_2022_nov_dec"]) >= 5,
        "train_2023_h1_minimum": len(periods["train_2023_h1"]) >= 16,
        "train_2023_h2_minimum": len(periods["train_2023_h2"]) >= 16,
        "train_maximum_month_share": _max_month_share(windows["train"]) <= 0.20,
        "test_total_minimum": len(windows["test"]) >= 35,
        "test_each_half_minimum": all(
            len(periods[name]) >= 14
            for name in ("test_2024_h1", "test_2024_h2")
        ),
        "test_each_quarter_minimum": all(
            len(quarters[name]) >= 5
            for name in ("2024Q1", "2024Q2", "2024Q3", "2024Q4")
        ),
        "test_maximum_month_share": _max_month_share(windows["test"]) <= 0.20,
        "eval_total_minimum": len(windows["eval"]) >= 35,
        "eval_each_half_minimum": all(
            len(periods[name]) >= 14
            for name in ("eval_2025_h1", "eval_2025_h2")
        ),
        "eval_each_quarter_minimum": all(
            len(quarters[name]) >= 5
            for name in ("2025Q1", "2025Q2", "2025Q3", "2025Q4")
        ),
        "eval_maximum_month_share": _max_month_share(windows["eval"]) <= 0.20,
        "forward_total_minimum": len(windows["forward"]) >= 18,
        "forward_2026_h1_minimum": len(periods["forward_2026_h1"]) >= 16,
        "forward_maximum_month_share": (
            _max_month_share(windows["forward"]) <= 0.28
        ),
        "missing_12h_source_buckets": bool(
            source["bucket_start_utc"]
            .diff()
            .dropna()
            .eq(pd.Timedelta(hours=12))
            .all()
        ),
    }
    side_minimums = {
        "train_long_minimum": side_counts["train"]["long"] >= 14,
        "train_short_minimum": side_counts["train"]["short"] >= 14,
        "test_long_minimum": side_counts["test"]["long"] >= 10,
        "test_short_minimum": side_counts["test"]["short"] >= 10,
        "eval_long_minimum": side_counts["eval"]["long"] >= 10,
        "eval_short_minimum": side_counts["eval"]["short"] >= 10,
        "forward_long_minimum": side_counts["forward"]["long"] >= 5,
        "forward_short_minimum": side_counts["forward"]["short"] >= 5,
    }
    checks.update(side_minimums)
    return {
        "passed": all(checks.values()),
        "control_name": control_name,
        "side_floor_contract": (
            "both-side floors required for primary and every control; fixed-side "
            "controls therefore cannot independently pass support"
        ),
        "waived_checks": [],
        "checks": checks,
        "counts": {
            "clock_total": int(len(clock)),
            **{name: int(len(window)) for name, window in windows.items()},
            **{name: int(len(period)) for name, period in periods.items()},
            **{name: int(len(period)) for name, period in quarters.items()},
        },
        "side_counts": side_counts,
        "maximum_month_share": {
            name: _max_month_share(window) for name, window in windows.items()
        },
    }


def control_support_summaries(
    controls: pd.DataFrame, source: pd.DataFrame
) -> dict[str, Any]:
    return {
        name: support_gate_summary(
            controls[controls["clock"] == name],
            source,
            control_name=name,
        )
        for name in CONTROL_NAMES
    }


def _format_clock(clock: pd.DataFrame) -> pd.DataFrame:
    out = clock.copy()
    for column in (
        "bucket_start_utc",
        "source_available_at_utc",
        "entry_time_utc",
        "exit_time_utc",
    ):
        out[column] = pd.to_datetime(out[column], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    return out[CLOCK_COLUMNS]


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    handle.close()
    return Path(handle.name)


def _write_clock(path: Path, clock: pd.DataFrame) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                _format_clock(clock).to_csv(
                    text,
                    index=False,
                    lineterminator="\n",
                    float_format="%.17g",
                )


def _frame_hash(clock: pd.DataFrame) -> str:
    return canonical_hash(_format_clock(clock).to_dict(orient="records"))


def _publish_new(temporary: Path, final: Path) -> None:
    os.link(temporary, final)


def _same_inode(left: Path, right: Path) -> bool:
    try:
        left_stat = left.stat()
        right_stat = right.stat()
    except FileNotFoundError:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (
        right_stat.st_dev,
        right_stat.st_ino,
    )


def build_support_artifacts(cfg: Config) -> dict[str, Any]:
    registration = validate_frozen_preregistration(cfg.preregistration)
    _validate_config(cfg, registration)
    source = validate_source_frame(load_source_frame(registration))
    features = build_features(source)
    primary = build_clock(features, mode="primary", clock="primary")
    controls = build_control_clocks(features, primary)
    gates = support_gate_summary(primary, source)
    control_gates = control_support_summaries(controls, source)

    output = _repository_path(cfg.output)
    primary_path = _repository_path(cfg.primary_clock)
    controls_path = _repository_path(cfg.control_clocks)
    output_tmp = _temporary_path(output)
    primary_tmp = _temporary_path(primary_path)
    controls_tmp = _temporary_path(controls_path)
    try:
        _write_clock(primary_tmp, primary)
        _write_clock(controls_tmp, controls)
        binding = _source_binding(registration)
        core = {
            "protocol_version": PROTOCOL_VERSION,
            "policy_id": POLICY_ID,
            "config": asdict(cfg),
            "support_builder": {
                "path": str(SUPPORT_BUILDER),
                "sha256": sha256_file(SUPPORT_BUILDER),
            },
            "preregistration": {
                "path": str(DEFAULT_PREREGISTRATION),
                "sha256": EXPECTED_PREREGISTRATION_FILE_SHA256,
                "manifest_hash": EXPECTED_PREREGISTRATION_MANIFEST_HASH,
                "policy_hash": EXPECTED_POLICY_HASH,
            },
            "source": {
                "path": binding["path"],
                "sha256": binding["sha256"],
                "bytes": binding["bytes"],
                "rows_read": int(len(source)),
                "columns": SOURCE_COLUMNS,
            },
            "feature_audit": {
                "feature_rows": int(len(features)),
                "base_valid_feature_rows": int(features["feature_valid"].sum()),
                "rank_ready_rows": int(features["rank_ready"].sum()),
                "lookback_valid_feature_buckets": 180,
                "minimum_prior_valid_feature_buckets": 120,
                "primary_thresholds": {
                    "magnitude_rank_minimum": 0.75,
                    "fullness_rank_minimum": 0.50,
                    "transport_impulse_nonzero_sign_confirmation": True,
                },
                "source_values_summarized": False,
            },
            "artifacts": {
                "primary_clock": {
                    "path": cfg.primary_clock,
                    "sha256": sha256_file(primary_tmp),
                    "frame_hash": _frame_hash(primary),
                    "rows": int(len(primary)),
                    "columns": CLOCK_COLUMNS,
                },
                "control_clocks": {
                    "path": cfg.control_clocks,
                    "sha256": sha256_file(controls_tmp),
                    "frame_hash": _frame_hash(controls),
                    "rows": int(len(controls)),
                    "columns": CLOCK_COLUMNS,
                    "clock_counts": {
                        name: int((controls["clock"] == name).sum())
                        for name in CONTROL_NAMES
                    },
                },
            },
            "support_gates": gates,
            "control_support_gates": control_gates,
            "outcome_boundary": OUTCOME_BOUNDARY,
            "performance_values_opened": False,
            "next_action": (
                "commit and hash-freeze the strict evaluator before outcomes"
                if gates["passed"]
                else "reject WCTR-288 without opening outcomes"
            ),
            "stopping_rule": (
                "reject permanently without outcomes on any primary support "
                "failure; no threshold, side, rank-window, support-floor, hold, "
                "latency, calendar, or clock repair"
            ),
        }
        artifact = {**core, "manifest_hash": canonical_hash(core)}
        output_tmp.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        published: list[tuple[Path, Path]] = []
        try:
            _publish_new(primary_tmp, primary_path)
            published.append((primary_tmp, primary_path))
            _publish_new(controls_tmp, controls_path)
            published.append((controls_tmp, controls_path))
            _publish_new(output_tmp, output)
            published.append((output_tmp, output))
        except BaseException:
            for temporary, final in reversed(published):
                if _same_inode(temporary, final):
                    final.unlink(missing_ok=True)
            raise
        return artifact
    finally:
        for path in (output_tmp, primary_tmp, controls_tmp):
            path.unlink(missing_ok=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--primary-clock", default=Config.primary_clock)
    parser.add_argument("--control-clocks", default=Config.control_clocks)
    parser.add_argument("--artifact-root", default=Config.artifact_root)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(
        json.dumps(
            build_support_artifacts(parse_args()),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
