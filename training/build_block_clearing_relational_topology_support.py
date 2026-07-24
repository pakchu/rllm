"""Build outcome-blind BCRT-72 source states, clocks, and support evidence."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_block_clearing_relational_topology as prereg


PROTOCOL_VERSION = "block_clearing_relational_topology_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/build_block_clearing_relational_topology_support.py"
)
TEST_PATH = Path(
    "tests/test_build_block_clearing_relational_topology_support.py"
)
IMPLEMENTATION_CONTRACT = Path(
    "docs/bcrt-source-support-implementation-contract-2026-07-24.md"
)
IMPLEMENTATION_CONTRACT_SHA256 = (
    "b0acdab2fa40b2988f5897f8b537bbebac70872d6a550fab9af0eb063a98658b"
)
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "322f91b41fce1aee06250a010d5a569557b83cc3f493ee3c47f5d6974aafe6a8"
)
PREREGISTRATION_MANIFEST_HASH = (
    "c9f08196f5a25dd05320a2c7cf3fbf951403d10f2362e67e2b0169b03fec194f"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "data/block_clearing_relational_topology_clocks_2020_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/block_clearing_relational_topology_support_2026-07-24.json"
)

SOURCE_START_SECONDS = 1_577_836_800
DEVELOPMENT_END_SECONDS = 1_672_531_200
SOURCE_END_SECONDS = 1_704_067_200
SOURCE_START = pd.Timestamp("2020-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2024-01-01T00:00:00Z")
WINDOWS = {
    "development": (
        SOURCE_START,
        pd.Timestamp("2023-01-01T00:00:00Z"),
    ),
    "train": (
        SOURCE_START,
        pd.Timestamp("2022-01-01T00:00:00Z"),
    ),
    "2020": (
        SOURCE_START,
        pd.Timestamp("2021-01-01T00:00:00Z"),
    ),
    "2021": (
        pd.Timestamp("2021-01-01T00:00:00Z"),
        pd.Timestamp("2022-01-01T00:00:00Z"),
    ),
    "2022": (
        pd.Timestamp("2022-01-01T00:00:00Z"),
        pd.Timestamp("2023-01-01T00:00:00Z"),
    ),
    "2023": (
        pd.Timestamp("2023-01-01T00:00:00Z"),
        SOURCE_END,
    ),
}
PAIR_TOKEN_COLUMNS = prereg.TOKEN_COLUMNS[:5]
LEADER_TOKEN_COLUMNS = ("high_leader", "low_leader")
CLOCK_COLUMNS = (
    "signal_id",
    "bucket_start",
    "signal_available_time",
    "entry_time",
    "exit_time",
    *prereg.TOKEN_COLUMNS,
)
INTERNAL_TIME_COLUMNS = (
    "bucket_start",
    "bucket_end",
    "anchor_timestamp",
    "anchor_mediantime",
    "confirmation_timestamp",
    "confirmation_mediantime",
    "signal_available_time",
    "entry_time",
    "exit_time",
)
PRIMITIVE_COLUMNS = tuple(name.lower() for name in prereg.PRIMITIVES)
RANK_COLUMNS = tuple(f"{name.lower()}_rank" for name in prereg.PRIMITIVES)
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_MANIFEST_OUTCOME_BOUNDARY = {
    "funding_rows_loaded": 0,
    "market_rows_loaded": 0,
    "outcome_rows_loaded": 0,
    "post_2023_source_rows_loaded": 0,
    "raw_mempool_responses_persisted": False,
    "return_or_pnl_fields": 0,
    "unrelated_mempool_metadata_persisted": False,
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


def _format_time(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("BCRT timestamp must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.microsecond or timestamp.nanosecond:
        raise RuntimeError("BCRT timestamp must be whole-second")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_protocol_committed() -> None:
    paths = (str(SCRIPT_PATH), str(TEST_PATH), str(IMPLEMENTATION_CONTRACT))
    tracked = _git_check("ls-files", "--error-unmatch", "--", *paths)
    if tracked.returncode:
        raise RuntimeError("BCRT source-support protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *paths)
    if clean.returncode:
        raise RuntimeError("BCRT source-support protocol differs from HEAD")


def validate_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("BCRT preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_manifest(payload)
    if payload != prereg.build_manifest():
        raise RuntimeError("BCRT preregistration differs from frozen builder")
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("BCRT preregistration manifest hash drift")
    boundary = payload["outcome_boundary"]
    for field in (
        "source_values_decoded",
        "bcrt_buckets_derived",
        "bcrt_primitive_or_rank_values_derived",
        "bcrt_token_rows_derived",
        "bcrt_opportunity_rows_derived",
        "market_rows_loaded",
        "funding_rows_loaded",
        "comparator_rows_decoded",
        "future_return_rows_loaded",
        "return_or_pnl_fields",
        "post_2023_rows_loaded",
        "model_labels_created",
        "model_training_runs",
    ):
        if boundary[field] != 0:
            raise RuntimeError(f"BCRT preregistration boundary opened: {field}")
    return payload


def _source_dependencies() -> dict[str, str]:
    return {
        str(IMPLEMENTATION_CONTRACT): IMPLEMENTATION_CONTRACT_SHA256,
        str(PREREGISTRATION): PREREGISTRATION_SHA256,
        prereg.BOUNDARY_DOCUMENT: prereg.BOUNDARY_DOCUMENT_SHA256,
        prereg.MECHANISM_DOCUMENT: prereg.MECHANISM_DOCUMENT_SHA256,
        prereg.SOURCE: prereg.SOURCE_SHA256,
        prereg.SOURCE_MANIFEST: prereg.SOURCE_MANIFEST_SHA256,
        prereg.SOURCE_BUILDER: prereg.SOURCE_BUILDER_SHA256,
        prereg.SOURCE_DECISION: prereg.SOURCE_DECISION_SHA256,
        prereg.REFERENCE: prereg.REFERENCE_SHA256,
    }


def _manifest_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }


def verify_pre_source_bindings(
    preregistration: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    audit: dict[str, dict[str, str]] = {}
    for path, expected in _source_dependencies().items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"BCRT frozen source binding changed: {path}")
        audit[path] = {"path": path, "sha256": actual}

    if prereg.sha256_csv_header(prereg.SOURCE) != prereg.SOURCE_HEADER_SHA256:
        raise RuntimeError("BCRT source header hash drift")
    if prereg.csv_header(prereg.SOURCE) != list(prereg.SOURCE_ALLOWLIST):
        raise RuntimeError("BCRT source allowlist/order drift")
    if prereg.sha256_csv_header(prereg.REFERENCE) != (
        prereg.REFERENCE_HEADER_SHA256
    ):
        raise RuntimeError("BCRT reference header hash drift")
    if prereg.csv_header(prereg.REFERENCE) != list(prereg.REFERENCE_ALLOWLIST):
        raise RuntimeError("BCRT reference allowlist/order drift")
    if preregistration["source_contract"]["allowlist"] != list(
        prereg.SOURCE_ALLOWLIST
    ):
        raise RuntimeError("BCRT preregistered source allowlist drift")

    manifest = json.loads(
        _path(prereg.SOURCE_MANIFEST).read_text(encoding="utf-8")
    )
    if not isinstance(manifest, dict):
        raise RuntimeError("BCRT source manifest is not an object")
    if manifest.get("manifest_hash") != canonical_hash(
        _manifest_core(manifest)
    ):
        raise RuntimeError("BCRT source manifest hash mismatch")
    if manifest.get("manifest_hash") != prereg.SOURCE_MANIFEST_HASH:
        raise RuntimeError("BCRT source manifest frozen hash drift")
    if manifest.get("protocol_version") != (
        "bitcoin_utxo_fee_block_stats_source_v1"
    ):
        raise RuntimeError("BCRT source manifest protocol drift")
    output = manifest.get("output")
    expected_output = {
        "bytes": 13_991_597,
        "columns": list(prereg.SOURCE_ALLOWLIST),
        "path": prereg.SOURCE,
        "sha256": prereg.SOURCE_SHA256,
    }
    if output != expected_output:
        raise RuntimeError("BCRT source output binding drift")
    config = manifest.get("config")
    if not isinstance(config, dict) or any(
        config.get(key) != value
        for key, value in {
            "start_height": 610_691,
            "end_height": 823_785,
            "end_timestamp_exclusive": SOURCE_END_SECONDS,
            "reference_block_summaries": prereg.REFERENCE,
            "reference_block_summaries_sha256": prereg.REFERENCE_SHA256,
        }.items()
    ):
        raise RuntimeError("BCRT source manifest config drift")
    if manifest.get("outcome_boundary") != SOURCE_MANIFEST_OUTCOME_BOUNDARY:
        raise RuntimeError("BCRT source manifest opened an outcome boundary")
    if manifest.get("source_builder") != {
        "path": prereg.SOURCE_BUILDER,
        "sha256": prereg.SOURCE_BUILDER_SHA256,
    }:
        raise RuntimeError("BCRT source builder binding drift")
    if manifest.get("source_decision") != {
        "path": prereg.SOURCE_DECISION,
        "sha256": prereg.SOURCE_DECISION_SHA256,
    }:
        raise RuntimeError("BCRT source decision binding drift")
    reference_audit = manifest.get("reference_audit")
    if not isinstance(reference_audit, dict) or any(
        reference_audit.get(key) != value
        for key, value in {
            "all_basic_fields_match_reference": True,
            "reference_path": prereg.REFERENCE,
            "reference_sha256": prereg.REFERENCE_SHA256,
            "rows_cross_checked": 213_095,
        }.items()
    ):
        raise RuntimeError("BCRT source reference audit drift")
    return audit


def _exact_integer_series(
    values: pd.Series,
    *,
    name: str,
) -> pd.Series:
    strings = values.astype("string")
    if strings.isna().any() or not strings.str.fullmatch(r"-?[0-9]+").all():
        raise RuntimeError(f"BCRT source {name} must be exact integers")
    numeric = pd.to_numeric(strings, errors="raise")
    if not pd.api.types.is_integer_dtype(numeric.dtype):
        raise RuntimeError(f"BCRT source {name} integer dtype drift")
    return numeric.astype("int64")


def validate_source_frame(
    frame: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    require_frozen_range: bool = True,
    cutoff_seconds: int = SOURCE_END_SECONDS,
) -> pd.DataFrame:
    if list(frame.columns) != list(prereg.SOURCE_ALLOWLIST):
        raise RuntimeError("BCRT source frame schema drift")
    if frame.empty:
        raise RuntimeError("BCRT source frame is empty")
    validated = frame.copy().reset_index(drop=True)
    integer_columns = [
        "height",
        "timestamp",
        "mediantime",
        "tx_count",
        "size",
        "weight",
        "total_fees",
        "total_inputs",
        "total_outputs",
        "utxo_set_change",
    ]
    for column in integer_columns:
        validated[column] = _exact_integer_series(
            validated[column],
            name=column,
        )
    for column in ("id", "previousblockhash"):
        validated[column] = validated[column].astype("string")
        if (
            validated[column].isna().any()
            or not validated[column].str.fullmatch(HEX_64.pattern).all()
        ):
            raise RuntimeError(f"BCRT source {column} must be lowercase hex")

    heights = validated["height"].to_numpy(dtype=np.int64)
    if require_frozen_range:
        expected = np.arange(610_691, 823_786, dtype=np.int64)
    else:
        expected = np.arange(heights[0], heights[0] + len(heights))
    if not np.array_equal(heights, expected):
        raise RuntimeError("BCRT source rows must be exact contiguous heights")
    if validated["height"].duplicated().any():
        raise RuntimeError("BCRT source heights must be unique")
    if validated["id"].duplicated().any():
        raise RuntimeError("BCRT source block ids must be unique")
    if not np.array_equal(
        validated["previousblockhash"].iloc[1:].to_numpy(),
        validated["id"].iloc[:-1].to_numpy(),
    ):
        raise RuntimeError("BCRT source parent linkage failed")
    if not (
        validated["total_outputs"] - validated["total_inputs"]
    ).eq(validated["utxo_set_change"]).all():
        raise RuntimeError("BCRT source UTXO identity failed")
    if validated["tx_count"].lt(1).any():
        raise RuntimeError("BCRT source tx_count must be at least one")
    if validated[["size", "weight"]].le(0).any().any():
        raise RuntimeError("BCRT source size and weight must be positive")
    if validated[
        ["total_fees", "total_inputs", "total_outputs"]
    ].lt(0).any().any():
        raise RuntimeError("BCRT source count or fee field is negative")
    if (
        validated["weight"].lt(validated["size"])
        | validated["weight"].gt(4 * validated["size"])
    ).any():
        raise RuntimeError("BCRT source size/weight invariant failed")
    if validated[["timestamp", "mediantime"]].le(0).any().any():
        raise RuntimeError("BCRT source clock must be positive")
    if validated[["timestamp", "mediantime"]].ge(cutoff_seconds).any().any():
        raise RuntimeError("BCRT source crossed the frozen cutoff")
    if not validated["mediantime"].is_monotonic_increasing:
        raise RuntimeError("BCRT source mediantime must be nondecreasing")

    if reference is not None:
        if list(reference.columns) != list(prereg.REFERENCE_ALLOWLIST):
            raise RuntimeError("BCRT reference frame schema drift")
        normalized_reference = reference.copy().reset_index(drop=True)
        for column in (
            "height",
            "timestamp",
            "mediantime",
            "tx_count",
            "size",
            "weight",
        ):
            normalized_reference[column] = _exact_integer_series(
                normalized_reference[column],
                name=f"reference_{column}",
            )
        for column in ("id", "previousblockhash"):
            normalized_reference[column] = normalized_reference[column].astype(
                "string"
            )
        if len(normalized_reference) != len(validated) or not validated.loc[
            :, list(prereg.REFERENCE_ALLOWLIST)
        ].equals(normalized_reference):
            raise RuntimeError("BCRT source/reference basic fields differ")
    return validated


def load_source_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(
        _path(prereg.SOURCE),
        usecols=list(prereg.SOURCE_ALLOWLIST),
        dtype="string",
    ).loc[:, list(prereg.SOURCE_ALLOWLIST)]
    reference = pd.read_csv(
        _path(prereg.REFERENCE),
        usecols=list(prereg.REFERENCE_ALLOWLIST),
        dtype="string",
    ).loc[:, list(prereg.REFERENCE_ALLOWLIST)]
    validated = validate_source_frame(source, reference=reference)
    return validated, reference


def _mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def primitive_values(members: pd.DataFrame) -> OrderedDict[str, float]:
    if members.empty:
        raise RuntimeError("BCRT primitive bucket has no members")
    n = len(members)
    weight = members["weight"].to_numpy(dtype=np.float64)
    size = members["size"].to_numpy(dtype=np.float64)
    tx = members["tx_count"].to_numpy(dtype=np.float64)
    fees = members["total_fees"].to_numpy(dtype=np.float64)
    utxo = members["utxo_set_change"].to_numpy(dtype=np.float64)
    total_weight = float(weight.sum())
    total_size = float(size.sum())
    total_tx = float(tx.sum())
    total_fees = float(fees.sum())
    total_utxo = float(utxo.sum())
    values = OrderedDict(
        (
            ("CADENCE", math.log(float(n))),
            (
                "UTILIZATION",
                math.log(
                    (total_weight + 1.0)
                    / (4_000_000.0 * float(n) + 1.0)
                ),
            ),
            (
                "PACKING",
                math.log((total_tx + 1.0) / (total_weight + 1.0)),
            ),
            (
                "FEE",
                math.log((total_fees + 1.0) / (total_weight + 1.0)),
            ),
            ("UTXO", total_utxo / (total_tx + 1.0)),
            (
                "WITNESS",
                (4.0 * total_size - total_weight) / (4.0 * total_size),
            ),
            ("LOAD_DISPERSION", _mad(weight / 4_000_000.0)),
            (
                "FEE_DISPERSION",
                _mad(np.log((fees + 1.0) / (weight + 1.0))),
            ),
        )
    )
    if tuple(values) != prereg.PRIMITIVES or not np.isfinite(
        np.asarray(list(values.values()), dtype=np.float64)
    ).all():
        raise RuntimeError("BCRT primitive construction drift")
    return values


def _bucket_state_digest(
    member_heights: Sequence[int],
    primitives: Mapping[str, float],
) -> str:
    return canonical_hash(
        {
            "member_heights": [int(value) for value in member_heights],
            "primitives_hex": {
                name: float(primitives[name]).hex()
                for name in prereg.PRIMITIVES
            },
        }
    )


def build_causal_buckets(
    frame: pd.DataFrame,
    *,
    start_seconds: int = SOURCE_START_SECONDS,
    end_seconds: int = SOURCE_END_SECONDS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if start_seconds % prereg.Policy().bucket_seconds or (
        end_seconds <= start_seconds
        or end_seconds % prereg.Policy().bucket_seconds
    ):
        raise RuntimeError("BCRT bucket range is not exact UTC half-days")
    source = frame.reset_index(drop=True)
    heights = source["height"].to_numpy(dtype=np.int64)
    timestamp = source["timestamp"].to_numpy(dtype=np.int64)
    mediantime = source["mediantime"].to_numpy(dtype=np.int64)
    if np.any(np.diff(heights) != 1):
        raise RuntimeError("BCRT bucket source heights are not contiguous")
    if np.any(np.diff(mediantime) < 0):
        raise RuntimeError("BCRT bucket source mediantime is not monotone")
    prefix_max_timestamp = np.maximum.accumulate(timestamp)
    prefix_max_mediantime = np.maximum.accumulate(mediantime)
    bucket_keys = (timestamp // prereg.Policy().bucket_seconds) * (
        prereg.Policy().bucket_seconds
    )
    bucket_index_lists: dict[int, list[int]] = {}
    for index, key in enumerate(bucket_keys):
        bucket_index_lists.setdefault(int(key), []).append(index)
    indexes_by_bucket = {
        key: np.asarray(indexes, dtype=np.int64)
        for key, indexes in bucket_index_lists.items()
    }

    records: list[dict[str, Any]] = []
    omitted_no_anchor = 0
    omitted_no_confirmation = 0
    late_members = 0
    later_append_rows_proved = 0
    development_replay_buckets = 0
    eval_replay_buckets = 0
    previous_anchor_height: int | None = None
    previous_confirmation_height: int | None = None
    previous_signal: int | None = None
    for bucket_start in range(
        start_seconds,
        end_seconds,
        prereg.Policy().bucket_seconds,
    ):
        bucket_end = bucket_start + prereg.Policy().bucket_seconds
        anchor_index = int(np.searchsorted(mediantime, bucket_end, side="left"))
        if anchor_index >= len(source):
            omitted_no_anchor += 1
            continue
        confirmation_index = (
            anchor_index + prereg.Policy().confirmation_blocks
        )
        if confirmation_index >= len(source):
            omitted_no_confirmation += 1
            continue
        anchor_height = int(heights[anchor_index])
        confirmation_height = int(heights[confirmation_index])
        if (
            previous_anchor_height is not None
            and anchor_height <= previous_anchor_height
        ):
            raise RuntimeError("BCRT anchor height did not increase")
        if (
            previous_confirmation_height is not None
            and confirmation_height <= previous_confirmation_height
        ):
            raise RuntimeError("BCRT confirmation height did not increase")

        bucket_indexes = indexes_by_bucket.get(
            bucket_start,
            np.empty(0, dtype=np.int64),
        )
        member_indexes = bucket_indexes[
            bucket_indexes <= confirmation_index
        ]
        later_indexes = bucket_indexes[
            bucket_indexes > confirmation_index
        ]
        if not len(member_indexes):
            raise RuntimeError("BCRT formed bucket has no prefix members")
        members = source.iloc[member_indexes]
        primitives = primitive_values(members)
        digest = _bucket_state_digest(
            members["height"].astype(int).tolist(),
            primitives,
        )

        replay_indexes = bucket_indexes[
            heights[bucket_indexes] <= confirmation_height
        ]
        replay_members = source.iloc[replay_indexes]
        replay_primitives = primitive_values(replay_members)
        replay_digest = _bucket_state_digest(
            replay_members["height"].astype(int).tolist(),
            replay_primitives,
        )
        if digest != replay_digest:
            raise RuntimeError("BCRT confirmation-prefix replay mismatch")
        if (
            confirmation_index + 1 < len(source)
            and source["height"].iloc[confirmation_index + 1 :].le(
                confirmation_height
            ).any()
        ):
            raise RuntimeError("BCRT later append can enter closed prefix")

        raw_available = (
            max(
                bucket_end,
                int(prefix_max_timestamp[confirmation_index]),
                int(prefix_max_mediantime[confirmation_index]),
            )
            + prereg.Policy().minimum_embargo_seconds
        )
        signal_available = prereg.ceil_5m(raw_available)
        entry = signal_available + (
            prereg.Policy().latency_bars * prereg.Policy().bar_seconds
        )
        exit_time = entry + (
            prereg.Policy().hold_bars * prereg.Policy().bar_seconds
        )
        if previous_signal is not None and signal_available < previous_signal:
            raise RuntimeError("BCRT signal availability is not monotone")

        record: dict[str, Any] = {
            "bucket_start_seconds": bucket_start,
            "bucket_end_seconds": bucket_end,
            "bucket_start": pd.Timestamp(bucket_start, unit="s", tz="UTC"),
            "bucket_end": pd.Timestamp(bucket_end, unit="s", tz="UTC"),
            "anchor_height": anchor_height,
            "confirmation_height": confirmation_height,
            "anchor_timestamp": pd.Timestamp(
                int(timestamp[anchor_index]),
                unit="s",
                tz="UTC",
            ),
            "anchor_mediantime": pd.Timestamp(
                int(mediantime[anchor_index]),
                unit="s",
                tz="UTC",
            ),
            "confirmation_timestamp": pd.Timestamp(
                int(timestamp[confirmation_index]),
                unit="s",
                tz="UTC",
            ),
            "confirmation_mediantime": pd.Timestamp(
                int(mediantime[confirmation_index]),
                unit="s",
                tz="UTC",
            ),
            "prefix_max_timestamp": int(
                prefix_max_timestamp[confirmation_index]
            ),
            "prefix_max_mediantime": int(
                prefix_max_mediantime[confirmation_index]
            ),
            "signal_available_time": pd.Timestamp(
                signal_available,
                unit="s",
                tz="UTC",
            ),
            "entry_time": pd.Timestamp(entry, unit="s", tz="UTC"),
            "exit_time": pd.Timestamp(exit_time, unit="s", tz="UTC"),
            "member_count": int(len(members)),
            "late_member_count": int(len(later_indexes)),
            "state_digest": digest,
            **{
                name.lower(): float(primitives[name])
                for name in prereg.PRIMITIVES
            },
        }
        records.append(record)
        previous_anchor_height = anchor_height
        previous_confirmation_height = confirmation_height
        previous_signal = signal_available
        late_members += int(len(later_indexes))
        later_append_rows_proved += len(source) - confirmation_index - 1
        if bucket_start < DEVELOPMENT_END_SECONDS:
            development_replay_buckets += 1
        else:
            eval_replay_buckets += 1

    buckets = pd.DataFrame(records)
    if buckets.empty:
        raise RuntimeError("BCRT source produced no formed buckets")
    return buckets, {
        "nominal_buckets": int(
            (end_seconds - start_seconds) // prereg.Policy().bucket_seconds
        ),
        "formed_buckets": int(len(buckets)),
        "omitted_no_anchor": int(omitted_no_anchor),
        "omitted_no_confirmation": int(omitted_no_confirmation),
        "late_backdated_members_excluded": int(late_members),
        "prefix_replay_buckets_checked": int(len(buckets)),
        "development_prefix_replay_buckets_checked": int(
            development_replay_buckets
        ),
        "eval_prefix_replay_buckets_report_only": int(eval_replay_buckets),
        "development_prefix_replay_passed": True,
        "eval_prefix_replay_report_only_passed": True,
        "later_append_rows_proved_excluded": int(
            later_append_rows_proved
        ),
        "prefix_replay_passed": True,
    }


def attach_strict_prior_ranks(buckets: pd.DataFrame) -> pd.DataFrame:
    ranked = buckets.sort_values(
        "bucket_start_seconds",
        kind="mergesort",
    ).reset_index(drop=True).copy()
    histories: dict[str, list[float]] = {
        name: [] for name in prereg.PRIMITIVES
    }
    result: dict[str, list[float]] = {name: [] for name in prereg.PRIMITIVES}
    for row in ranked.itertuples(index=False):
        current = {
            name: float(getattr(row, name.lower()))
            for name in prereg.PRIMITIVES
        }
        if not np.isfinite(
            np.asarray(list(current.values()), dtype=np.float64)
        ).all():
            raise RuntimeError("BCRT bucket primitive is non-finite")
        ready = all(
            len(histories[name])
            >= prereg.Policy().rank_minimum_prior_buckets
            for name in prereg.PRIMITIVES
        )
        for name in prereg.PRIMITIVES:
            if ready:
                rank = prereg.strict_prior_midrank(
                    current[name],
                    histories[name],
                )
            else:
                rank = np.nan
            result[name].append(rank)
        for name in prereg.PRIMITIVES:
            histories[name].append(current[name])
    for name, values in result.items():
        ranked[f"{name.lower()}_rank"] = np.asarray(
            values,
            dtype=np.float64,
        )
    ranked["rank_complete"] = ranked.loc[:, RANK_COLUMNS].notna().all(axis=1)
    return ranked


def _rank_mapping(row: Any) -> OrderedDict[str, float]:
    return OrderedDict(
        (
            name,
            float(getattr(row, f"{name.lower()}_rank")),
        )
        for name in prereg.PRIMITIVES
    )


def relational_tokens(
    current: Mapping[str, float],
    previous: Mapping[str, float],
) -> dict[str, str]:
    pairs = (
        (
            "cadence_utilization",
            "CADENCE",
            "UTILIZATION",
            "CADENCE_LEADS",
            "UTILIZATION_LEADS",
        ),
        (
            "utilization_fee",
            "UTILIZATION",
            "FEE",
            "UTILIZATION_LEADS",
            "FEE_LEADS",
        ),
        (
            "packing_witness",
            "PACKING",
            "WITNESS",
            "PACKING_LEADS",
            "WITNESS_LEADS",
        ),
        (
            "utxo_fee",
            "UTXO",
            "FEE",
            "UTXO_LEADS",
            "FEE_LEADS",
        ),
        (
            "load_fee_dispersion",
            "LOAD_DISPERSION",
            "FEE_DISPERSION",
            "LOAD_WIDER",
            "FEE_WIDER",
        ),
    )
    pair_values: OrderedDict[str, str] = OrderedDict()
    relation_scores: list[int] = []
    for column, left, right, left_token, right_token in pairs:
        token = prereg.pair_relation(
            current[left],
            current[right],
            left_token=left_token,
            right_token=right_token,
        )
        pair_values[column] = token
        relation_scores.append(
            1 if token == left_token else -1 if token == right_token else 0
        )
    current_high = prereg.extreme_leader(current, highest=True)
    current_low = prereg.extreme_leader(current, highest=False)
    previous_high = prereg.extreme_leader(previous, highest=True)
    previous_low = prereg.extreme_leader(previous, highest=False)
    tokens = OrderedDict(
        (
            *pair_values.items(),
            ("high_leader", current_high),
            ("low_leader", current_low),
            ("rank_breadth", prereg.rank_breadth(current)),
            (
                "extreme_occupancy",
                prereg.extreme_occupancy(current),
            ),
            (
                "relation_breadth",
                prereg.relation_breadth(relation_scores),
            ),
            (
                "order_transition",
                prereg.order_transition(current, previous),
            ),
            (
                "leader_transition",
                prereg.leader_transition(
                    current_high,
                    current_low,
                    previous_high,
                    previous_low,
                ),
            ),
        )
    )
    return prereg.validate_tokens(tokens)


def signal_id(row: Mapping[str, Any]) -> str:
    payload = {
        "policy_id": prereg.POLICY_ID,
        "bucket_start": _format_time(row["bucket_start"]),
        "entry_time": _format_time(row["entry_time"]),
        "tokens": {
            column: str(row[column]) for column in prereg.TOKEN_COLUMNS
        },
    }
    return f"BCRT-{canonical_hash(payload)[:24]}"


def build_token_candidates(
    ranked: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    complete = ranked.loc[ranked["rank_complete"]].reset_index(drop=True)
    records: list[dict[str, Any]] = []
    previous: OrderedDict[str, float] | None = None
    for row in complete.itertuples(index=False):
        current = _rank_mapping(row)
        if previous is not None:
            tokens = relational_tokens(current, previous)
            record = {
                "bucket_start": pd.Timestamp(row.bucket_start),
                "bucket_end": pd.Timestamp(row.bucket_end),
                "anchor_timestamp": pd.Timestamp(row.anchor_timestamp),
                "anchor_mediantime": pd.Timestamp(row.anchor_mediantime),
                "confirmation_timestamp": pd.Timestamp(
                    row.confirmation_timestamp
                ),
                "confirmation_mediantime": pd.Timestamp(
                    row.confirmation_mediantime
                ),
                "signal_available_time": pd.Timestamp(
                    row.signal_available_time
                ),
                "entry_time": pd.Timestamp(row.entry_time),
                "exit_time": pd.Timestamp(row.exit_time),
                **tokens,
            }
            record["signal_id"] = signal_id(record)
            records.append(record)
        previous = current
    columns = (
        "signal_id",
        *INTERNAL_TIME_COLUMNS,
        *prereg.TOKEN_COLUMNS,
    )
    return pd.DataFrame(records, columns=columns), {
        "formed_buckets": int(len(ranked)),
        "rank_complete_states": int(len(complete)),
        "first_rank_complete_predecessor_only": int(bool(len(complete))),
        "token_ready_states": int(len(records)),
    }


def reserve_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    ordered = candidates.sort_values(
        ["entry_time", "exit_time", "signal_id"],
        kind="mergesort",
    ).reset_index(drop=True).copy()
    previous_exit: pd.Timestamp | None = None
    reserved: list[bool] = []
    for row in ordered.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if entry.tzinfo is None or exit_time.tzinfo is None or exit_time <= entry:
            raise RuntimeError("BCRT reservation interval is invalid")
        accepted = previous_exit is None or entry >= previous_exit
        reserved.append(accepted)
        if accepted:
            previous_exit = exit_time
    ordered["reserved"] = reserved
    return ordered


def _split_for_bucket(bucket_start: pd.Timestamp) -> str | None:
    for name in ("2020", "2021", "2022", "2023"):
        start, end = WINDOWS[name]
        if start <= bucket_start < end:
            return name
    return None


def split_contained(row: Mapping[str, Any]) -> tuple[bool, str | None]:
    bucket_start = pd.Timestamp(row["bucket_start"])
    split = _split_for_bucket(bucket_start)
    if split is None:
        return False, None
    start, end = WINDOWS[split]
    point_fields = (
        "anchor_timestamp",
        "anchor_mediantime",
        "confirmation_timestamp",
        "confirmation_mediantime",
        "signal_available_time",
        "entry_time",
    )
    contained = (
        bucket_start >= start
        and pd.Timestamp(row["bucket_end"]) <= end
        and all(
            start <= pd.Timestamp(row[field]) < end for field in point_fields
        )
        and pd.Timestamp(row["exit_time"]) < end
    )
    return bool(contained), split


def eligible_clock(
    reserved_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    annotated = reserved_candidates.copy()
    containment = [
        split_contained(row)
        for row in annotated.to_dict(orient="records")
    ]
    annotated["split_contained"] = [item[0] for item in containment]
    annotated["split"] = [item[1] for item in containment]
    emitted = annotated.loc[
        annotated["reserved"] & annotated["split_contained"],
        list(CLOCK_COLUMNS),
    ].reset_index(drop=True)
    return emitted, {
        "token_ready": int(len(annotated)),
        "globally_reserved": int(annotated["reserved"].sum()),
        "overlap_suppressed": int((~annotated["reserved"]).sum()),
        "split_suppressed_after_reservation": int(
            (annotated["reserved"] & ~annotated["split_contained"]).sum()
        ),
        "emitted": int(len(emitted)),
    }


def _window(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = WINDOWS[name]
    entry = pd.to_datetime(rows["entry_time"], utc=True)
    return rows.loc[entry.ge(start) & entry.lt(end)].reset_index(drop=True)


def _month_counts(rows: pd.DataFrame) -> Counter[str]:
    return Counter(
        pd.Timestamp(value).strftime("%Y-%m")
        for value in rows["entry_time"]
    )


def _maximum_gap_days(rows: pd.DataFrame) -> int | None:
    if len(rows) < 2:
        return None
    dates = [
        pd.Timestamp(value).date()
        for value in rows["entry_time"].sort_values(kind="mergesort")
    ]
    return max(
        (current - previous).days
        for previous, current in zip(dates, dates[1:])
    )


def clock_stats(rows: pd.DataFrame) -> dict[str, Any]:
    total = len(rows)
    months = _month_counts(rows)
    return {
        "events": int(total),
        "first_entry": (
            _format_time(rows["entry_time"].min()) if total else None
        ),
        "last_exit": (
            _format_time(rows["exit_time"].max()) if total else None
        ),
        "active_months": int(len(months)),
        "maximum_month_share": (
            float(max(months.values()) / total) if total else None
        ),
        "maximum_gap_days": _maximum_gap_days(rows),
    }


def _partition_counts(rows: pd.DataFrame, year: int) -> dict[str, int]:
    entry = pd.to_datetime(rows["entry_time"], utc=True)
    boundaries = (
        (f"{year}_h1", f"{year}-01-01", f"{year}-07-01"),
        (f"{year}_h2", f"{year}-07-01", f"{year + 1}-01-01"),
        (f"{year}_q1", f"{year}-01-01", f"{year}-04-01"),
        (f"{year}_q2", f"{year}-04-01", f"{year}-07-01"),
        (f"{year}_q3", f"{year}-07-01", f"{year}-10-01"),
        (f"{year}_q4", f"{year}-10-01", f"{year + 1}-01-01"),
    )
    return {
        name: int(
            (
                entry.ge(pd.Timestamp(start, tz="UTC"))
                & entry.lt(pd.Timestamp(end, tz="UTC"))
            ).sum()
        )
        for name, start, end in boundaries
    }


def token_report(rows: pd.DataFrame) -> dict[str, Any]:
    total = len(rows)
    counts: dict[str, dict[str, int]] = {}
    shares: dict[str, dict[str, float]] = {}
    for column in prereg.TOKEN_COLUMNS:
        observed = rows[column].astype(str).value_counts().sort_index()
        counts[column] = {
            str(key): int(value) for key, value in observed.items()
        }
        shares[column] = {
            str(key): float(value / total)
            for key, value in observed.items()
        } if total else {}
    signatures = (
        rows.loc[:, list(prereg.TOKEN_COLUMNS)]
        .astype(str)
        .agg("|".join, axis=1)
        .value_counts()
    ) if total else pd.Series(dtype=np.int64)
    return {
        "events": int(total),
        "counts": counts,
        "shares": shares,
        "distinct_signatures": int(len(signatures)),
        "maximum_exact_signature_share": (
            float(signatures.max() / total) if total else None
        ),
    }


def token_support_checks(
    rows: pd.DataFrame,
    *,
    split: str,
    gate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    report = token_report(rows)
    shares = report["shares"]
    checks: dict[str, bool] = {}
    for column in PAIR_TOKEN_COLUMNS:
        vocabulary = prereg.TOKEN_VOCABULARY[column]
        checks[f"{split}:{column}:each_value_share_min"] = all(
            shares[column].get(value, 0.0)
            >= gate["pair_each_value_share_min"]
            for value in vocabulary
        )
        checks[f"{split}:{column}:max_value_share"] = bool(
            shares[column]
            and max(shares[column].values()) <= gate["pair_max_value_share"]
        )
    for column in LEADER_TOKEN_COLUMNS:
        values = rows[column].astype(str)
        non_tie = values[values.ne("TIE")]
        non_tie_counts = non_tie.value_counts()
        checks[f"{split}:{column}:nontie_distinct_min"] = (
            int(non_tie.nunique()) >= gate["leader_nontie_distinct_min"]
        )
        checks[f"{split}:{column}:max_nontie_share"] = bool(
            len(non_tie)
            and float(non_tie_counts.max() / len(non_tie))
            <= gate["leader_max_nontie_share"]
        )
        checks[f"{split}:{column}:tie_share_max"] = bool(
            len(values)
            and float(values.eq("TIE").sum() / len(values))
            <= gate["leader_tie_share_max"]
        )
    share_contracts = (
        (
            "rank_breadth",
            "rank_breadth_each_share_min",
            "rank_breadth_max_share",
        ),
        (
            "extreme_occupancy",
            "extreme_occupancy_each_share_min",
            "extreme_occupancy_max_share",
        ),
        (
            "relation_breadth",
            "relation_breadth_each_share_min",
            "relation_breadth_max_share",
        ),
        (
            "order_transition",
            "order_transition_each_share_min",
            "order_transition_max_share",
        ),
    )
    for column, minimum_key, maximum_key in share_contracts:
        vocabulary = prereg.TOKEN_VOCABULARY[column]
        checks[f"{split}:{column}:each_value_share_min"] = all(
            shares[column].get(value, 0.0) >= gate[minimum_key]
            for value in vocabulary
        )
        checks[f"{split}:{column}:max_value_share"] = bool(
            shares[column]
            and max(shares[column].values()) <= gate[maximum_key]
        )
    transition = rows["leader_transition"].astype(str)
    transition_counts = transition.value_counts()
    checks[f"{split}:leader_transition:distinct_min"] = (
        int(transition.nunique()) >= gate["leader_transition_distinct_min"]
    )
    checks[f"{split}:leader_transition:max_share"] = bool(
        len(transition)
        and float(transition_counts.max() / len(transition))
        <= gate["leader_transition_max_share"]
    )
    checks[f"{split}:maximum_exact_signature_share"] = bool(
        report["maximum_exact_signature_share"] is not None
        and report["maximum_exact_signature_share"]
        <= gate["max_exact_signature_share"]
    )
    checks[f"{split}:all_tokens_valid"] = all(
        value in prereg.TOKEN_VOCABULARY[column]
        for column in prereg.TOKEN_COLUMNS
        for value in rows[column].astype(str)
    )
    return report, checks


def _timing_integrity(rows: pd.DataFrame) -> bool:
    if rows.empty:
        return False
    signal = pd.to_datetime(rows["signal_available_time"], utc=True)
    entry = pd.to_datetime(rows["entry_time"], utc=True)
    exit_time = pd.to_datetime(rows["exit_time"], utc=True)
    return bool(
        signal.dt.second.eq(0).all()
        and signal.dt.microsecond.eq(0).all()
        and signal.map(lambda value: value.minute % 5 == 0).all()
        and entry.sub(signal).eq(pd.Timedelta(minutes=5)).all()
        and exit_time.sub(entry).eq(pd.Timedelta(hours=6)).all()
    )


def _reservation_integrity(rows: pd.DataFrame) -> bool:
    if rows.empty:
        return False
    ordered = rows.sort_values("entry_time", kind="mergesort")
    entries = pd.to_datetime(ordered["entry_time"], utc=True).iloc[1:]
    exits = pd.to_datetime(ordered["exit_time"], utc=True).iloc[:-1]
    return bool(
        len(ordered) == 1
        or np.all(entries.to_numpy() >= exits.to_numpy())
    )


def support_checks(
    rows: pd.DataFrame,
    *,
    development_replay_passed: bool,
    eval_replay_report_only_passed: bool = True,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, int]],
    dict[str, Any],
    dict[str, bool],
    dict[str, bool],
    dict[str, Any],
]:
    contract = prereg.build_manifest()["source_support_gates"]
    incidence_gate = contract["development_incidence"]
    token_gate = contract["token_support"]
    statistics = {
        name: clock_stats(_window(rows, name)) for name in WINDOWS
    }
    partitions = {
        str(year): _partition_counts(rows, year)
        for year in range(2020, 2024)
    }
    development_rows = _window(rows, "development")
    source_checks = {
        "development_2020_2022_min": (
            statistics["development"]["events"]
            >= incidence_gate["development_2020_2022_min"]
        ),
        "train_2020_2021_min": (
            statistics["train"]["events"]
            >= incidence_gate["train_2020_2021_min"]
        ),
        "year_2020_min": (
            statistics["2020"]["events"] >= incidence_gate["year_2020_min"]
        ),
        "each_year_2021_2022_min": all(
            statistics[str(year)]["events"]
            >= incidence_gate["each_year_2021_2022_min"]
            for year in (2021, 2022)
        ),
        "year_2020_active_months_min": (
            statistics["2020"]["active_months"]
            >= incidence_gate["year_2020_active_months_min"]
        ),
        "each_year_2021_2022_active_months_min": all(
            statistics[str(year)]["active_months"]
            >= incidence_gate["each_year_2021_2022_active_months_min"]
            for year in (2021, 2022)
        ),
        "each_half_2021_2022_min": all(
            partitions[str(year)][f"{year}_h{half}"]
            >= incidence_gate["each_half_2021_2022_min"]
            for year in (2021, 2022)
            for half in (1, 2)
        ),
        "each_quarter_2021_2022_min": all(
            partitions[str(year)][f"{year}_q{quarter}"]
            >= incidence_gate["each_quarter_2021_2022_min"]
            for year in (2021, 2022)
            for quarter in range(1, 5)
        ),
        "year_2020_max_month_share": bool(
            statistics["2020"]["maximum_month_share"] is not None
            and statistics["2020"]["maximum_month_share"]
            <= incidence_gate["year_2020_max_month_share"]
        ),
        "each_year_2021_2022_max_month_share": all(
            statistics[str(year)]["maximum_month_share"] is not None
            and statistics[str(year)]["maximum_month_share"]
            <= incidence_gate["each_year_2021_2022_max_month_share"]
            for year in (2021, 2022)
        ),
        "max_entry_gap_days_2020_2022": bool(
            statistics["development"]["maximum_gap_days"] is not None
            and statistics["development"]["maximum_gap_days"]
            <= incidence_gate["max_entry_gap_days_2020_2022"]
        ),
        "development_prefix_replay": bool(development_replay_passed),
        "development_timing_integrity": _timing_integrity(development_rows),
        "development_global_nonoverlap": _reservation_integrity(
            development_rows
        ),
        "development_clock_schema_exact": (
            list(development_rows.columns) == list(CLOCK_COLUMNS)
        ),
    }

    train = _window(rows, "train")
    selection = _window(rows, "2022")
    evaluation = _window(rows, "2023")
    train_report, train_checks = token_support_checks(
        train,
        split="train",
        gate=token_gate,
    )
    selection_report, selection_checks = token_support_checks(
        selection,
        split="2022",
        gate=token_gate,
    )
    token_checks = {**train_checks, **selection_checks}
    train_values = {
        column: set(train[column].astype(str))
        for column in prereg.TOKEN_COLUMNS
    }
    for column in prereg.TOKEN_COLUMNS:
        downstream = set(selection[column].astype(str))
        token_checks[f"2022:{column}:seen_in_train"] = (
            downstream <= train_values[column]
        )
    eval_report = token_report(evaluation)
    eval_report["calendar"] = statistics["2023"]
    eval_report["partition_counts"] = partitions["2023"]
    eval_report["train_vocabulary_coverage"] = {
        column: sorted(
            set(evaluation[column].astype(str)) - train_values[column]
        )
        for column in prereg.TOKEN_COLUMNS
    }
    eval_report["operational_validity_report"] = {
        "prefix_replay_passed": bool(eval_replay_report_only_passed),
        "timing_integrity": (
            _timing_integrity(evaluation) if len(evaluation) else None
        ),
        "global_nonoverlap": (
            _reservation_integrity(evaluation) if len(evaluation) else None
        ),
        "clock_schema_exact": (
            list(evaluation.columns) == list(CLOCK_COLUMNS)
        ),
    }
    eval_report["boolean_gate"] = False
    eval_report["may_authorize_continue_retire_repair_or_selection"] = False
    return (
        statistics,
        partitions,
        {"train": train_report, "2022": selection_report},
        source_checks,
        token_checks,
        eval_report,
    )


def first_failure(
    source_checks: Mapping[str, bool],
    token_checks: Mapping[str, bool],
    *,
    artifact_eligible: bool,
) -> tuple[str, str | None]:
    for name, passed in source_checks.items():
        if not passed:
            return "source_support", name
    for name, passed in token_checks.items():
        if not passed:
            return "token_support", name
    if not artifact_eligible:
        return "artifact_eligibility", "synthetic_or_injected_build"
    return "none", None


def deterministic_clock_bytes(rows: pd.DataFrame) -> bytes:
    if list(rows.columns) != list(CLOCK_COLUMNS):
        raise RuntimeError("BCRT clock schema drift")
    serialized = rows.sort_values(
        ["entry_time", "signal_id"],
        kind="mergesort",
    ).copy()
    for column in (
        "bucket_start",
        "signal_available_time",
        "entry_time",
        "exit_time",
    ):
        serialized[column] = serialized[column].map(_format_time)
    text = serialized.to_csv(
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


def _frame_hash(rows: pd.DataFrame) -> str:
    ordered = rows.sort_values(
        ["entry_time", "signal_id"],
        kind="mergesort",
    ).copy()
    for column in (
        "bucket_start",
        "signal_available_time",
        "entry_time",
        "exit_time",
    ):
        ordered[column] = ordered[column].map(_format_time)
    return canonical_hash(ordered.loc[:, list(CLOCK_COLUMNS)].to_dict("records"))


def _core_payload(
    rows: pd.DataFrame,
    *,
    source_audit: Mapping[str, Any],
    bucket_audit: Mapping[str, Any],
    feature_funnel: Mapping[str, Any],
    reservation_funnel: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    clock_bytes: bytes,
    artifact_eligible: bool,
) -> dict[str, Any]:
    (
        statistics,
        partitions,
        token_reports,
        source_checks,
        token_checks,
        eval_report,
    ) = support_checks(
        rows,
        development_replay_passed=bool(
            bucket_audit.get("development_prefix_replay_passed")
        ),
        eval_replay_report_only_passed=bool(
            bucket_audit.get("eval_prefix_replay_report_only_passed")
        ),
    )
    source_passed = all(source_checks.values())
    token_passed = bool(source_passed and all(token_checks.values()))
    first_stage, first_check = first_failure(
        source_checks,
        token_checks,
        artifact_eligible=artifact_eligible,
    )
    if not source_passed or not token_passed:
        decision = "retire_BCRT_72_unchanged_before_market_outcomes"
    elif not artifact_eligible:
        decision = "synthetic_build_cannot_authorize_market_outcomes"
    else:
        decision = "advance_to_frozen_cheap_baseline_evaluator"
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.POLICY_ID,
        "artifact_eligible": artifact_eligible,
        "source_incidence_opened": artifact_eligible,
        "outcomes_opened": False,
        "market_loaded": False,
        "funding_loaded": False,
        "comparators_opened": False,
        "post_2023_loaded": False,
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
            "contract": str(IMPLEMENTATION_CONTRACT),
            "contract_sha256": IMPLEMENTATION_CONTRACT_SHA256,
        },
        "source_audit": dict(source_audit),
        "bucket_audit": dict(bucket_audit),
        "feature_funnel": dict(feature_funnel),
        "reservation_funnel": dict(reservation_funnel),
        "clock_statistics": statistics,
        "calendar_partition_counts": partitions,
        "development_token_report": token_reports,
        "source_support_checks": source_checks,
        "source_support_passed": source_passed,
        "token_support_checks": token_checks,
        "token_support_passed": token_passed,
        "eval_source_report_only": eval_report,
        "first_failing_stage": first_stage,
        "first_failing_check": first_check,
        "clock": {
            "path": str(DEFAULT_CLOCK_OUTPUT),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "frame_hash": _frame_hash(rows),
            "rows": int(len(rows)),
            "columns": list(CLOCK_COLUMNS),
        },
        "decision": decision,
        "authorized_next_stage": (
            "freeze_cheap_baseline_and_economic_evaluator"
            if token_passed and artifact_eligible
            else None
        ),
        "outcome_boundary": {
            "source_rows_decoded": int(
                source_audit.get("source_rows_decoded", 0)
            ),
            "reference_rows_decoded": int(
                source_audit.get("reference_rows_decoded", 0)
            ),
            "bcrt_buckets_derived": int(
                bucket_audit.get("formed_buckets", 0)
            ),
            "bcrt_primitive_rows_derived": int(
                bucket_audit.get("formed_buckets", 0)
            ),
            "bcrt_rank_rows_derived": int(
                feature_funnel.get("rank_complete_states", 0)
            ),
            "bcrt_token_rows_derived": int(
                feature_funnel.get("token_ready_states", 0)
            ),
            "bcrt_reserved_rows_derived": int(
                reservation_funnel.get("globally_reserved", 0)
            ),
            "BTC_market_rows_decoded": 0,
            "funding_rows_decoded": 0,
            "comparator_rows_decoded": 0,
            "future_return_rows_decoded": 0,
            "return_or_PnL_fields_decoded": 0,
            "PnL_CAGR_MDD_values_decoded": 0,
            "post_2023_rows_decoded": 0,
            "model_labels_created": 0,
            "model_training_runs": 0,
            "network_calls": 0,
        },
        "binding_manifest_hash": preregistration["manifest_hash"],
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def build_support_from_frame(
    frame: pd.DataFrame,
    *,
    start_seconds: int,
    end_seconds: int,
) -> tuple[dict[str, Any], bytes]:
    validated = validate_source_frame(
        frame,
        require_frozen_range=False,
        cutoff_seconds=end_seconds + 10 * 365 * 86_400,
    )
    buckets, bucket_audit = build_causal_buckets(
        validated,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    ranked = attach_strict_prior_ranks(buckets)
    candidates, feature_funnel = build_token_candidates(ranked)
    reserved = reserve_candidates(candidates)
    rows, reservation_funnel = eligible_clock(reserved)
    clock_bytes = deterministic_clock_bytes(rows)
    payload = validate_preregistration()
    report = _core_payload(
        rows,
        source_audit={
            "source_rows_decoded": int(len(validated)),
            "reference_rows_decoded": 0,
            "synthetic_or_injected": True,
        },
        bucket_audit=bucket_audit,
        feature_funnel=feature_funnel,
        reservation_funnel=reservation_funnel,
        preregistration=payload,
        clock_bytes=clock_bytes,
        artifact_eligible=False,
    )
    return report, clock_bytes


def build_real_support_payload() -> tuple[dict[str, Any], bytes]:
    _assert_protocol_committed()
    payload = validate_preregistration()
    bindings = verify_pre_source_bindings(payload)
    source, reference = load_source_frames()
    buckets, bucket_audit = build_causal_buckets(source)
    ranked = attach_strict_prior_ranks(buckets)
    candidates, feature_funnel = build_token_candidates(ranked)
    reserved = reserve_candidates(candidates)
    rows, reservation_funnel = eligible_clock(reserved)
    clock_bytes = deterministic_clock_bytes(rows)
    source_audit = {
        "source": {
            "path": prereg.SOURCE,
            "sha256": prereg.SOURCE_SHA256,
            "header_sha256": prereg.SOURCE_HEADER_SHA256,
            "allowlist": list(prereg.SOURCE_ALLOWLIST),
        },
        "reference": {
            "path": prereg.REFERENCE,
            "sha256": prereg.REFERENCE_SHA256,
            "header_sha256": prereg.REFERENCE_HEADER_SHA256,
            "allowlist": list(prereg.REFERENCE_ALLOWLIST),
        },
        "source_rows_decoded": int(len(source)),
        "reference_rows_decoded": int(len(reference)),
        "pre_source_bindings": bindings,
        "synthetic_or_injected": False,
        "source_validation_passed": True,
        "reference_equality_passed": True,
    }
    return (
        _core_payload(
            rows,
            source_audit=source_audit,
            bucket_audit=bucket_audit,
            feature_funnel=feature_funnel,
            reservation_funnel=reservation_funnel,
            preregistration=payload,
            clock_bytes=clock_bytes,
            artifact_eligible=True,
        ),
        clock_bytes,
    )


def _write_once(path: str | Path, payload: bytes) -> str:
    output = _path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError(f"BCRT noncanonical existing artifact: {path}")
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
                raise RuntimeError(f"BCRT artifact race drift: {path}")
            return "verified_existing"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def write_support(
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> dict[str, Any]:
    if Path(report_output) != DEFAULT_REPORT_OUTPUT:
        raise RuntimeError("BCRT real report output path is frozen")
    if Path(clock_output) != DEFAULT_CLOCK_OUTPUT:
        raise RuntimeError("BCRT real clock output path is frozen")
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
        "token_support_passed": report["token_support_passed"],
        "decision": report["decision"],
        "manifest_hash": report["manifest_hash"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-output",
        default=str(DEFAULT_REPORT_OUTPUT),
    )
    parser.add_argument(
        "--clock-output",
        default=str(DEFAULT_CLOCK_OUTPUT),
    )
    args = parser.parse_args()
    result = write_support(args.report_output, args.clock_output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
