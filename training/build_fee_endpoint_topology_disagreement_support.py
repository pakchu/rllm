"""Build outcome-blind FETD-288 packet features and support clocks.

This stage opens exactly one hash-bound confirmed-ledger CSV snapshot.  It
derives only source features, primary incidence, and frozen control clocks.  It
publishes aggregate support evidence plus sealed clock hashes, never event rows
or feature values, and never opens BTC market data, funding, returns, PnL, or
post-2023 source rows.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import preregister_fee_endpoint_topology_disagreement as prereg


POLICY_ID = "FETD-288"
PROTOCOL_VERSION = "fee_endpoint_topology_disagreement_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = Path(
    "results/fee_endpoint_topology_disagreement_preregistration_2026-07-20.json"
)
DEFAULT_OUTPUT = Path(
    "results/fee_endpoint_topology_disagreement_support_2026-07-20.json"
)
SUPPORT_BUILDER = Path(
    "training/build_fee_endpoint_topology_disagreement_support.py"
)
EXPECTED_PREREGISTRATION_FILE_SHA256 = (
    "2de820b6f78d0cd566f2750f91bfca8c092795ab93b81121326bdb067247e285"
)
EXPECTED_PREREGISTRATION_MANIFEST_HASH = (
    "8e2042c5439ca6a7b73cbc05eb2c8f0600edc622833bc33afaabb12e178e0fbe"
)
EXPECTED_POLICY_HASH = (
    "7bd47d38c93c6507ce75bab455f3b3e7efee69a803f56bc1ab78ebcb85d9434e"
)
EXPECTED_PREREGISTRATION_SOURCE_SHA256 = (
    "ae1329f0cd124787d822096e56dc3bc3ed05ccd2f2f6f0cb86f47e5cd766c413"
)

SOURCE_COLUMNS = list(prereg.SOURCE_COLUMNS)
INTEGER_COLUMNS = [
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
SIGNAL_SOURCE_COLUMNS = {
    "height",
    "id",
    "previousblockhash",
    "timestamp",
    "weight",
    "total_fees",
    "total_inputs",
    "total_outputs",
}
FORBIDDEN_SIGNAL_SOURCE_COLUMNS = {
    "mediantime",
    "tx_count",
    "size",
    "utxo_set_change",
}
FORBIDDEN_OUTCOME_COLUMNS = {
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
    "liquidation",
    "order_book",
}
FEATURE_COLUMNS = [
    "fee_pressure",
    "endpoint_density",
    "fee_transport",
    "endpoint_transport",
    "strain_magnitude",
    "strain_rank",
    "fee_magnitude_rank",
    "endpoint_magnitude_rank",
]
CLOCK_COLUMNS = [
    "policy_id",
    "clock",
    "window",
    "packet_id",
    "packet_start_height",
    "packet_end_height",
    "confirmation_end_height",
    "source_available_at_utc",
    "entry_time_utc",
    "exit_time_utc",
    "side",
    *FEATURE_COLUMNS,
]
CONTROL_NAMES = [
    "direction_flip",
    "fee_only",
    "endpoint_only",
    "same_direction",
    "constant_long_same_clock",
    "constant_short_same_clock",
    "stale_14_packets",
    "month_side_stratified_random_clock",
    "one_bar_delayed_entry",
]
WINDOWS = {
    "train": (
        pd.Timestamp("2021-01-01T00:00:00Z"),
        pd.Timestamp("2023-01-01T00:00:00Z"),
    ),
    "selection": (
        pd.Timestamp("2023-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:00:00Z"),
    ),
}
OUTCOME_BOUNDARY = {
    "source_values_read": True,
    "source_feature_rows_derived": True,
    "signal_incidence_rows_derived": True,
    "post_2023_source_rows_loaded": 0,
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "premium_or_oi_rows_loaded": 0,
    "liquidation_or_order_book_rows_loaded": 0,
    "return_rows_loaded": 0,
    "market_values_read": 0,
    "funding_values_read": 0,
    "return_or_pnl_fields": 0,
}
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Config:
    preregistration: str = str(DEFAULT_PREREGISTRATION)
    output: str = str(DEFAULT_OUTPUT)
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
            raise RuntimeError(f"FETD support JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _expected_source_binding() -> dict[str, Any]:
    return {
        "path": str(prereg.SOURCE_MANIFEST),
        "sha256": prereg.EXPECTED_SOURCE_MANIFEST_FILE_SHA256,
        "manifest_hash": prereg.EXPECTED_SOURCE_MANIFEST_HASH,
        "protocol_version": prereg.SOURCE_PROTOCOL_VERSION,
        "source_output": {
            "path": str(prereg.EXPECTED_SOURCE_OUTPUT),
            "sha256": prereg.EXPECTED_SOURCE_OUTPUT_SHA256,
            "bytes": prereg.EXPECTED_SOURCE_OUTPUT_BYTES,
            "columns": list(prereg.SOURCE_COLUMNS),
        },
        "source_origin_decision": {
            "path": str(prereg.SOURCE_ORIGIN_DECISION),
            "sha256": prereg.SOURCE_ORIGIN_DECISION_SHA256,
        },
        "source_builder": {
            "path": str(prereg.SOURCE_BUILDER),
            "sha256": prereg.SOURCE_BUILDER_SHA256,
        },
        "source_audit": {
            "expected_rows": prereg.FROZEN_ROWS,
            "observed_rows": prereg.FROZEN_ROWS,
            "start_height": prereg.FROZEN_START_HEIGHT,
            "end_height": prereg.FROZEN_END_HEIGHT,
            "latest_eligible_packet_end": prereg.FROZEN_END_HEIGHT - 6,
            "height_links_checked": prereg.FROZEN_ROWS - 1,
            "end_timestamp_exclusive": prereg.FROZEN_END_TIMESTAMP_EXCLUSIVE,
            "complete_inclusive_height_range": True,
            "unique_block_hashes": True,
            "all_rows_pre_cutoff": True,
            "utxo_identity_checked": True,
        },
        "reference_audit": {
            "reference_path": str(prereg.REFERENCE_SOURCE),
            "reference_sha256": prereg.REFERENCE_SOURCE_SHA256,
            "rows_cross_checked": prereg.FROZEN_ROWS,
            "columns_cross_checked": list(prereg.REFERENCE_COLUMNS),
            "all_basic_fields_match_reference": True,
        },
        "outcome_boundary": dict(prereg.SOURCE_OUTCOME_BOUNDARY),
        "data_use": prereg.EXPECTED_DATA_USE,
    }


def validate_frozen_preregistration(path: str | Path) -> dict[str, Any]:
    artifact_path = _repository_path(path)
    if artifact_path != _repository_path(DEFAULT_PREREGISTRATION):
        raise RuntimeError("FETD preregistration path differs from frozen artifact")
    artifact_bytes = artifact_path.read_bytes()
    if (
        hashlib.sha256(artifact_bytes).hexdigest()
        != EXPECTED_PREREGISTRATION_FILE_SHA256
    ):
        raise RuntimeError("FETD preregistration file SHA drift")
    artifact = json.loads(
        artifact_bytes.decode("utf-8"), object_pairs_hook=_unique_object
    )
    if not isinstance(artifact, dict):
        raise RuntimeError("FETD preregistration must be a JSON object")
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    if canonical_hash(core) != artifact.get("manifest_hash"):
        raise RuntimeError("FETD preregistration canonical hash mismatch")
    if artifact.get("protocol_version") != prereg.PROTOCOL_VERSION:
        raise RuntimeError("FETD preregistration protocol drift")
    if artifact.get("manifest_hash") != EXPECTED_PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("FETD preregistration manifest hash drift")
    if artifact.get("policy_hash") != EXPECTED_POLICY_HASH:
        raise RuntimeError("FETD policy hash drift")
    if canonical_hash(artifact.get("policy")) != EXPECTED_POLICY_HASH:
        raise RuntimeError("FETD policy content drift")
    if artifact.get("policy") != prereg.policy():
        raise RuntimeError("FETD policy singleton drift")
    if artifact.get("policy_id") != POLICY_ID:
        raise RuntimeError("FETD policy id drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("FETD preregistration opened outcomes")
    if artifact.get("outcome_boundary") != prereg.PREREGISTRATION_OUTCOME_BOUNDARY:
        raise RuntimeError("FETD preregistration outcome boundary drift")

    expected_source = {
        "path": str(prereg.PREREGISTRATION_SOURCE),
        "sha256": EXPECTED_PREREGISTRATION_SOURCE_SHA256,
    }
    if artifact.get("preregistration_source") != expected_source:
        raise RuntimeError("FETD preregistration source binding drift")
    if (
        sha256_file(prereg.PREREGISTRATION_SOURCE)
        != EXPECTED_PREREGISTRATION_SOURCE_SHA256
    ):
        raise RuntimeError("FETD preregistration source file drift")
    expected_decision = {
        "path": str(prereg.MECHANISM_DECISION),
        "sha256": prereg.MECHANISM_DECISION_SHA256,
    }
    if artifact.get("mechanism_decision") != expected_decision:
        raise RuntimeError("FETD mechanism decision binding drift")
    if sha256_file(prereg.MECHANISM_DECISION) != prereg.MECHANISM_DECISION_SHA256:
        raise RuntimeError("FETD mechanism decision file drift")
    expected_document = {
        "path": str(prereg.PREREGISTRATION_DOCUMENT),
        "sha256": prereg.PREREGISTRATION_DOCUMENT_SHA256,
    }
    if artifact.get("preregistration_document") != expected_document:
        raise RuntimeError("FETD preregistration document binding drift")
    if (
        sha256_file(prereg.PREREGISTRATION_DOCUMENT)
        != prereg.PREREGISTRATION_DOCUMENT_SHA256
    ):
        raise RuntimeError("FETD preregistration document file drift")
    if artifact.get("source_manifest") != _expected_source_binding():
        raise RuntimeError("FETD frozen source metadata binding drift")
    return artifact


def _source_binding(registration: dict[str, Any]) -> dict[str, Any]:
    source = registration["source_manifest"].get("source_output")
    if not isinstance(source, dict):
        raise RuntimeError("FETD source output binding missing")
    return source


def _validate_config(cfg: Config, registration: dict[str, Any]) -> None:
    output = _repository_path(cfg.output)
    if not str(cfg.output).endswith(".json"):
        raise ValueError("FETD support output must be JSON")
    protected = {
        _repository_path(cfg.preregistration),
        _repository_path(prereg.SOURCE_MANIFEST),
        _repository_path(_source_binding(registration)["path"]),
        _repository_path(prereg.REFERENCE_SOURCE),
        _repository_path(prereg.SOURCE_ORIGIN_DECISION),
        _repository_path(prereg.MECHANISM_DECISION),
        _repository_path(prereg.SOURCE_BUILDER),
        _repository_path(prereg.PREREGISTRATION_SOURCE),
        _repository_path(prereg.PREREGISTRATION_DOCUMENT),
        _repository_path(SUPPORT_BUILDER),
    }
    if output in protected:
        raise ValueError("FETD support output aliases a frozen input")
    artifact_root = _repository_path(cfg.artifact_root)
    if not output.is_relative_to(artifact_root):
        raise ValueError("FETD support output must stay under the artifact root")
    if output.exists():
        raise FileExistsError("FETD support artifact is immutable")


def validate_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != SOURCE_COLUMNS:
        raise RuntimeError("FETD source frame schema drift")
    if frame.empty:
        raise RuntimeError("FETD source frame is empty")
    forbidden = FORBIDDEN_OUTCOME_COLUMNS.intersection(
        {str(column).lower() for column in frame.columns}
    )
    if forbidden:
        raise RuntimeError(f"FETD source contains outcome-like columns: {forbidden!r}")

    out = frame.copy()
    for column in INTEGER_COLUMNS:
        values = pd.to_numeric(out[column], errors="raise")
        if values.isna().any() or (values % 1).ne(0).any():
            raise RuntimeError(f"FETD source {column} must contain exact integers")
        out[column] = values.astype(np.int64)
    out["id"] = out["id"].astype(str)
    out["previousblockhash"] = out["previousblockhash"].astype(str)
    out = out.reset_index(drop=True)

    heights = out["height"].to_numpy(np.int64)
    expected = np.arange(heights[0], heights[0] + len(out), dtype=np.int64)
    if not np.array_equal(heights, expected):
        raise RuntimeError("FETD source rows must have exact contiguous heights")
    if out["id"].duplicated().any():
        raise RuntimeError("FETD source block ids must be unique")
    if not out["id"].map(lambda value: bool(HASH_RE.fullmatch(value))).all():
        raise RuntimeError("FETD source block ids must be lowercase 64-hex")
    if not out["previousblockhash"].map(
        lambda value: bool(HASH_RE.fullmatch(value))
    ).all():
        raise RuntimeError("FETD source previous hashes must be lowercase 64-hex")
    if not np.array_equal(
        out["previousblockhash"].iloc[1:].to_numpy(),
        out["id"].iloc[:-1].to_numpy(),
    ):
        raise RuntimeError("FETD source hash-chain linkage failed")

    positive = ["timestamp", "mediantime", "tx_count", "size", "weight"]
    if out[positive].le(0).any().any():
        raise RuntimeError("FETD source contains non-positive required fields")
    nonnegative = ["total_fees", "total_inputs", "total_outputs"]
    if out[nonnegative].lt(0).any().any():
        raise RuntimeError("FETD source contains negative fee/endpoint counts")
    if not (out["total_outputs"] - out["total_inputs"]).eq(
        out["utxo_set_change"]
    ).all():
        raise RuntimeError("FETD source UTXO identity failed")
    size = out["size"].to_numpy(np.int64)
    weight = out["weight"].to_numpy(np.int64)
    if np.any(size > 4_000_000) or np.any(weight > 4_000_000):
        raise RuntimeError("FETD source size/weight exceeds frozen bounds")
    if np.any(weight < size) or np.any(weight > 4 * size):
        raise RuntimeError("FETD source violates BIP141 size/weight bounds")
    if np.any(out["timestamp"].to_numpy(np.int64) < out["mediantime"].to_numpy(np.int64)):
        raise RuntimeError("FETD source block timestamp precedes current MTP")
    return out


def load_source_frame(registration: dict[str, Any]) -> pd.DataFrame:
    binding = _source_binding(registration)
    path = _repository_path(binding["path"])
    compressed = path.read_bytes()
    if len(compressed) != binding["bytes"]:
        raise RuntimeError("FETD source byte-size drift")
    if hashlib.sha256(compressed).hexdigest() != binding["sha256"]:
        raise RuntimeError("FETD source SHA drift")
    frame = pd.read_csv(
        io.BytesIO(compressed),
        compression="gzip",
        dtype={"id": "string", "previousblockhash": "string"},
    )
    if frame.columns.tolist() != SOURCE_COLUMNS:
        raise RuntimeError("FETD source CSV header drift")
    return validate_source_frame(frame)


def _exact_source_audit(source: pd.DataFrame) -> dict[str, Any]:
    checks = {
        "exact_rows": len(source) == prereg.FROZEN_ROWS,
        "exact_start_height": int(source["height"].iloc[0])
        == prereg.FROZEN_START_HEIGHT,
        "exact_end_height": int(source["height"].iloc[-1])
        == prereg.FROZEN_END_HEIGHT,
        "all_rows_pre_cutoff": bool(
            source["timestamp"].lt(prereg.FROZEN_END_TIMESTAMP_EXCLUSIVE).all()
        ),
        "unique_block_hashes": not source["id"].duplicated().any(),
        "hash_chain_contiguous": bool(
            np.array_equal(
                source["previousblockhash"].iloc[1:].to_numpy(),
                source["id"].iloc[:-1].to_numpy(),
            )
        ),
        "utxo_identity": bool(
            (source["total_outputs"] - source["total_inputs"])
            .eq(source["utxo_set_change"])
            .all()
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def build_packets(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = validate_source_frame(source)
    frame = frame.copy()
    frame["packet_id"] = frame["height"] // 72
    rows: list[dict[str, Any]] = []
    partial_packet_ids: list[int] = []
    for packet_id, group in frame.groupby("packet_id", sort=True, observed=True):
        group = group.sort_values("height", kind="stable")
        expected_start = int(packet_id) * 72
        expected_heights = np.arange(expected_start, expected_start + 72)
        complete = bool(
            len(group) == 72
            and np.array_equal(group["height"].to_numpy(np.int64), expected_heights)
        )
        if not complete:
            partial_packet_ids.append(int(packet_id))
            continue
        start_index = int(group.index[0])
        end_index = int(group.index[-1])
        if end_index - start_index != 71:
            raise RuntimeError("FETD complete packet rows are not contiguous")
        confirmation_index = end_index + 6
        confirmation_contained = confirmation_index < len(frame)
        if not confirmation_contained:
            raise RuntimeError("FETD complete packet lacks six successors")
        confirmation_end_height = int(frame.at[confirmation_index, "height"])
        if confirmation_end_height != expected_start + 77:
            raise RuntimeError("FETD packet confirmation height drift")
        availability_timestamp = int(
            frame.loc[start_index:confirmation_index, "timestamp"].max()
        ) + 172_800
        available = pd.Timestamp(availability_timestamp, unit="s", tz="UTC")
        entry = available.ceil("5min") + pd.Timedelta(minutes=5)
        total_weight = int(group["weight"].sum())
        total_fees = int(group["total_fees"].sum())
        total_endpoints = int(
            (group["total_inputs"] + group["total_outputs"]).sum()
        )
        valid = bool(total_weight > 0 and total_fees > 0 and total_endpoints > 0)
        fee_pressure = math.log(total_fees / total_weight) if valid else math.nan
        endpoint_density = (
            math.log(total_endpoints / total_weight) if valid else math.nan
        )
        valid = bool(
            valid and math.isfinite(fee_pressure) and math.isfinite(endpoint_density)
        )
        rows.append(
            {
                "packet_id": int(packet_id),
                "packet_start_height": expected_start,
                "packet_end_height": expected_start + 71,
                "confirmation_end_height": confirmation_end_height,
                "packet_valid": valid,
                "fee_pressure": fee_pressure,
                "endpoint_density": endpoint_density,
                "source_available_at_utc": available,
                "entry_time_utc": entry,
                "exit_time_utc": entry + pd.Timedelta(hours=24),
            }
        )
    packets = pd.DataFrame.from_records(rows)
    if packets.empty:
        raise RuntimeError("FETD source produced no complete packets")
    packets = packets.sort_values("packet_id", kind="stable").reset_index(drop=True)
    packet_ids = packets["packet_id"].to_numpy(np.int64)
    expected_ids = np.arange(packet_ids[0], packet_ids[0] + len(packet_ids))
    if not np.array_equal(packet_ids, expected_ids):
        raise RuntimeError("FETD complete packet ids are not consecutive")
    source_first_id = int(frame["packet_id"].iloc[0])
    source_last_id = int(frame["packet_id"].iloc[-1])
    allowed_partial = {source_first_id, source_last_id}
    if not set(partial_packet_ids).issubset(allowed_partial):
        raise RuntimeError("FETD source contains an incomplete interior packet")
    audit = {
        "complete_packets": int(len(packets)),
        "first_complete_packet_id": int(packet_ids[0]),
        "last_complete_packet_id": int(packet_ids[-1]),
        "first_complete_packet_start_height": int(
            packets["packet_start_height"].iloc[0]
        ),
        "last_complete_packet_end_height": int(
            packets["packet_end_height"].iloc[-1]
        ),
        "partial_edge_packet_ids": partial_packet_ids,
        "all_complete_packets_have_72_blocks": True,
        "complete_packet_ids_consecutive": True,
        "all_confirmation_blocks_contained": True,
    }
    return packets, audit


def strict_prior_midrank(current: float, prior: Iterable[float]) -> float:
    values = list(prior)
    if not values:
        raise ValueError("FETD strict-prior midrank requires prior values")
    less = sum(value < current for value in values)
    equal = sum(value == current for value in values)
    return (less + 0.5 * equal) / len(values)


def build_features(
    packets: pd.DataFrame,
    *,
    lookback: int = 180,
    minimum_prior: int = 120,
) -> pd.DataFrame:
    if lookback < minimum_prior or minimum_prior <= 0:
        raise ValueError("FETD rank lookback/minimum contract is invalid")
    required = {
        "packet_id",
        "packet_start_height",
        "packet_end_height",
        "confirmation_end_height",
        "packet_valid",
        "fee_pressure",
        "endpoint_density",
        "source_available_at_utc",
        "entry_time_utc",
        "exit_time_utc",
    }
    if set(packets.columns) != required:
        raise RuntimeError("FETD packet frame schema drift")
    ordered = packets.sort_values("packet_id", kind="stable").reset_index(drop=True)
    if ordered["packet_id"].duplicated().any():
        raise RuntimeError("FETD packet ids must be unique")
    if not ordered["packet_id"].diff().dropna().eq(1).all():
        raise RuntimeError("FETD feature packets must be consecutive")

    records: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for index, row in ordered.iterrows():
        record = row.to_dict()
        feature_values = {column: math.nan for column in FEATURE_COLUMNS[2:]}
        base_valid = False
        if index >= 2:
            prior_two = ordered.iloc[index - 2 : index + 1]
            ingredient_availability = pd.to_datetime(
                prior_two["source_available_at_utc"], utc=True
            )
            current_availability = pd.Timestamp(row["source_available_at_utc"])
            if ingredient_availability.gt(current_availability).any():
                raise RuntimeError(
                    "FETD feature ingredient is unavailable at current packet clock"
                )
            consecutive = prior_two["packet_id"].diff().dropna().eq(1).all()
            all_valid = prior_two["packet_valid"].eq(True).all()
            if consecutive and all_valid:
                fee_transport = float(row["fee_pressure"]) - float(
                    ordered.at[index - 2, "fee_pressure"]
                )
                endpoint_transport = float(row["endpoint_density"]) - float(
                    ordered.at[index - 2, "endpoint_density"]
                )
                strain = abs(fee_transport * endpoint_transport)
                base_valid = bool(
                    math.isfinite(fee_transport)
                    and math.isfinite(endpoint_transport)
                    and math.isfinite(strain)
                )
                if base_valid:
                    feature_values.update(
                        {
                            "fee_transport": fee_transport,
                            "endpoint_transport": endpoint_transport,
                            "strain_magnitude": strain,
                        }
                    )

        available = pd.Timestamp(row["source_available_at_utc"])
        rank_ready = False
        if base_valid:
            causal_prior = [
                item for item in history if item["available_at"] < available
            ][-lookback:]
            if len(causal_prior) >= minimum_prior:
                feature_values["strain_rank"] = strict_prior_midrank(
                    feature_values["strain_magnitude"],
                    (item["strain_magnitude"] for item in causal_prior),
                )
                feature_values["fee_magnitude_rank"] = strict_prior_midrank(
                    abs(feature_values["fee_transport"]),
                    (item["fee_magnitude"] for item in causal_prior),
                )
                feature_values["endpoint_magnitude_rank"] = strict_prior_midrank(
                    abs(feature_values["endpoint_transport"]),
                    (item["endpoint_magnitude"] for item in causal_prior),
                )
                rank_ready = all(
                    math.isfinite(feature_values[column])
                    for column in (
                        "strain_rank",
                        "fee_magnitude_rank",
                        "endpoint_magnitude_rank",
                    )
                )
            history.append(
                {
                    "available_at": available,
                    "strain_magnitude": feature_values["strain_magnitude"],
                    "fee_magnitude": abs(feature_values["fee_transport"]),
                    "endpoint_magnitude": abs(
                        feature_values["endpoint_transport"]
                    ),
                }
            )
        record.update(feature_values)
        record.update({"feature_valid": base_valid, "rank_ready": rank_ready})
        records.append(record)
    return pd.DataFrame.from_records(records)


def _window_name(entry: pd.Timestamp, exit_time: pd.Timestamp) -> str | None:
    for name, (start, end) in WINDOWS.items():
        if start <= entry and exit_time <= end:
            return name
    return None


def _candidate_side(row: Any, mode: str) -> int:
    if not bool(row.rank_ready):
        return 0
    fee = float(row.fee_transport)
    endpoint = float(row.endpoint_transport)
    if mode == "primary":
        eligible = bool(fee * endpoint < 0.0 and float(row.strain_rank) >= 0.75)
        side_value = -fee
    elif mode == "fee_only":
        eligible = bool(fee != 0.0 and float(row.fee_magnitude_rank) >= 0.75)
        side_value = -fee
    elif mode == "endpoint_only":
        eligible = bool(
            endpoint != 0.0 and float(row.endpoint_magnitude_rank) >= 0.75
        )
        side_value = endpoint
    elif mode == "same_direction":
        eligible = bool(fee * endpoint > 0.0 and float(row.strain_rank) >= 0.75)
        side_value = endpoint
    elif mode == "rank_ready":
        eligible = True
        side_value = 1.0
    else:
        raise ValueError(f"unknown FETD candidate mode {mode!r}")
    if not eligible or side_value == 0.0:
        return 0
    return 1 if side_value > 0.0 else -1


def _clock_record(row: Any, *, clock: str, window: str, side: int) -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "clock": clock,
        "window": window,
        "packet_id": int(row.packet_id),
        "packet_start_height": int(row.packet_start_height),
        "packet_end_height": int(row.packet_end_height),
        "confirmation_end_height": int(row.confirmation_end_height),
        "source_available_at_utc": row.source_available_at_utc,
        "entry_time_utc": row.entry_time_utc,
        "exit_time_utc": row.exit_time_utc,
        "side": int(side),
        **{column: float(getattr(row, column)) for column in FEATURE_COLUMNS},
    }


def build_clock(features: pd.DataFrame, *, mode: str, clock: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    prior_exit: dict[str, pd.Timestamp] = {}
    ordered = features.sort_values(["entry_time_utc", "packet_id"], kind="stable")
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
    shifted = features[["packet_id", *FEATURE_COLUMNS, "rank_ready"]].shift(14)
    exact_lag = (stale["packet_id"] - shifted["packet_id"]).eq(14)
    for column in [*FEATURE_COLUMNS, "rank_ready"]:
        stale[column] = shifted[column]
    stale["rank_ready"] = shifted["rank_ready"].eq(True) & exact_lag
    return build_clock(stale, mode="primary", clock="stale_14_packets")


def _random_key(window: str, month: str, entry: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(entry).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = (
        f"20260720|{window}|{month}|{timestamp}".encode("ascii")
    )
    return hashlib.sha256(payload).hexdigest()


def _random_clock(features: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    pool_records: list[dict[str, Any]] = []
    ordered = features.sort_values(["entry_time_utc", "packet_id"], kind="stable")
    for row in ordered.itertuples(index=False):
        if _candidate_side(row, "rank_ready") == 0:
            continue
        entry = pd.Timestamp(row.entry_time_utc)
        exit_time = pd.Timestamp(row.exit_time_utc)
        window = _window_name(entry, exit_time)
        if window is None:
            continue
        pool_records.append(
            _clock_record(row, clock="random_pool", window=window, side=1)
        )
    baseline = pd.DataFrame.from_records(pool_records, columns=CLOCK_COLUMNS)
    primary = primary.copy()
    primary["_month"] = primary["entry_time_utc"].dt.strftime("%Y-%m")
    baseline["_month"] = baseline["entry_time_utc"].dt.strftime("%Y-%m")
    records: list[pd.DataFrame] = []
    for (window, month), target in primary.groupby(
        ["window", "_month"], sort=True, observed=True
    ):
        candidates = baseline[
            (baseline["window"] == window) & (baseline["_month"] == month)
        ].copy()
        if len(candidates) < len(target):
            raise RuntimeError("FETD random control month pool is too small")
        candidates["_key"] = [
            _random_key(str(window), str(month), entry)
            for entry in candidates["entry_time_utc"]
        ]
        sampled = candidates.sort_values(
            ["_key", "entry_time_utc", "packet_id"], kind="stable"
        ).head(len(target)).copy()
        long_count = int((target["side"] == 1).sum())
        sampled["side"] = [-1] * len(sampled)
        sampled.iloc[:long_count, sampled.columns.get_loc("side")] = 1
        sampled["clock"] = "month_side_stratified_random_clock"
        records.append(sampled[CLOCK_COLUMNS])
    return (
        pd.concat(records, ignore_index=True)[CLOCK_COLUMNS]
        .sort_values(["entry_time_utc", "packet_id"], kind="stable")
        .reset_index(drop=True)
    )


def _delayed_clock(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    delayed = primary.copy()
    delayed["clock"] = "one_bar_delayed_entry"
    delayed["entry_time_utc"] += pd.Timedelta(minutes=5)
    delayed["exit_time_utc"] += pd.Timedelta(minutes=5)
    contained = pd.Series(
        [
            _window_name(row.entry_time_utc, row.exit_time_utc) == row.window
            for row in delayed.itertuples(index=False)
        ],
        index=delayed.index,
        dtype=bool,
    )
    dropped = delayed.loc[~contained]
    dropped_counts = {
        name: int((dropped["window"] == name).sum()) for name in WINDOWS
    }
    return (
        delayed.loc[contained, CLOCK_COLUMNS].reset_index(drop=True),
        dropped_counts,
    )


def build_control_clocks(
    features: pd.DataFrame, primary: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    delayed, delayed_dropped = _delayed_clock(primary)
    controls = [
        _same_clock(primary, "direction_flip", -primary["side"].astype(int)),
        build_clock(features, mode="fee_only", clock="fee_only"),
        build_clock(features, mode="endpoint_only", clock="endpoint_only"),
        build_clock(features, mode="same_direction", clock="same_direction"),
        _same_clock(primary, "constant_long_same_clock", [1] * len(primary)),
        _same_clock(primary, "constant_short_same_clock", [-1] * len(primary)),
        _stale_clock(features),
        _random_clock(features, primary),
        delayed,
    ]
    nonempty = [control for control in controls if not control.empty]
    combined = (
        pd.concat(nonempty, ignore_index=True)
        if nonempty
        else pd.DataFrame(columns=CLOCK_COLUMNS)
    )
    observed = set(combined["clock"].unique())
    if not observed.issubset(set(CONTROL_NAMES)):
        raise RuntimeError("FETD control family drift")
    combined = combined.sort_values(
        ["clock", "entry_time_utc", "packet_id"], kind="stable"
    ).reset_index(drop=True)
    return combined, delayed_dropped


def _subset(clock: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
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
    packet_audit: dict[str, Any],
    *,
    delayed_dropped: dict[str, int] | None = None,
    control_name: str | None = None,
) -> dict[str, Any]:
    windows = {name: clock[clock["window"] == name] for name in WINDOWS}
    periods = {
        "2021": _subset(clock, "2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
        "2022": _subset(clock, "2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
        "2021H1": _subset(clock, "2021-01-01T00:00:00Z", "2021-07-01T00:00:00Z"),
        "2021H2": _subset(clock, "2021-07-01T00:00:00Z", "2022-01-01T00:00:00Z"),
        "2022H1": _subset(clock, "2022-01-01T00:00:00Z", "2022-07-01T00:00:00Z"),
        "2022H2": _subset(clock, "2022-07-01T00:00:00Z", "2023-01-01T00:00:00Z"),
        "2023H1": _subset(clock, "2023-01-01T00:00:00Z", "2023-07-01T00:00:00Z"),
        "2023H2": _subset(clock, "2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    }
    quarters = {
        name: _subset(clock, start, end)
        for name, start, end in (
            ("2023Q1", "2023-01-01T00:00:00Z", "2023-04-01T00:00:00Z"),
            ("2023Q2", "2023-04-01T00:00:00Z", "2023-07-01T00:00:00Z"),
            ("2023Q3", "2023-07-01T00:00:00Z", "2023-10-01T00:00:00Z"),
            ("2023Q4", "2023-10-01T00:00:00Z", "2024-01-01T00:00:00Z"),
        )
    }
    side_counts = {
        name: {
            "long": int((window["side"] == 1).sum()),
            "short": int((window["side"] == -1).sum()),
        }
        for name, window in windows.items()
    }
    year_side = {
        year: {
            "long": int((periods[year]["side"] == 1).sum()),
            "short": int((periods[year]["side"] == -1).sum()),
        }
        for year in ("2021", "2022")
    }
    half_side = {
        half: {
            "long": int((periods[half]["side"] == 1).sum()),
            "short": int((periods[half]["side"] == -1).sum()),
        }
        for half in ("2023H1", "2023H2")
    }
    checks = {
        "source_complete_packet_count": packet_audit.get("complete_packets")
        == 2_959,
        "source_first_complete_packet_start": packet_audit.get(
            "first_complete_packet_start_height"
        )
        == 610_704,
        "source_last_complete_packet_end": packet_audit.get(
            "last_complete_packet_end_height"
        )
        == 823_751,
        "source_complete_packets_have_72_blocks": packet_audit.get(
            "all_complete_packets_have_72_blocks"
        )
        is True,
        "source_complete_packet_ids_consecutive": packet_audit.get(
            "complete_packet_ids_consecutive"
        )
        is True,
        "source_confirmation_blocks_contained": packet_audit.get(
            "all_confirmation_blocks_contained"
        )
        is True,
        "train_total_minimum": len(windows["train"]) >= 80,
        "train_each_year_minimum": all(
            len(periods[year]) >= 32 for year in ("2021", "2022")
        ),
        "train_long_minimum": side_counts["train"]["long"] >= 25,
        "train_short_minimum": side_counts["train"]["short"] >= 25,
        "train_each_side_each_year_minimum": all(
            year_side[year][side] >= 10
            for year in ("2021", "2022")
            for side in ("long", "short")
        ),
        "train_each_half_year_minimum": all(
            len(periods[half]) >= 14
            for half in ("2021H1", "2021H2", "2022H1", "2022H2")
        ),
        "train_maximum_month_share": _max_month_share(windows["train"]) <= 0.15,
        "selection_total_minimum": len(windows["selection"]) >= 35,
        "selection_long_minimum": side_counts["selection"]["long"] >= 12,
        "selection_short_minimum": side_counts["selection"]["short"] >= 12,
        "selection_each_half_minimum": all(
            len(periods[half]) >= 14 for half in ("2023H1", "2023H2")
        ),
        "selection_each_side_each_half_minimum": all(
            half_side[half][side] >= 5
            for half in ("2023H1", "2023H2")
            for side in ("long", "short")
        ),
        "selection_each_quarter_minimum": all(
            len(quarters[name]) >= 6 for name in quarters
        ),
        "selection_maximum_month_share": (
            _max_month_share(windows["selection"]) <= 0.20
        ),
        "delayed_entry_split_edge_counts_recorded": delayed_dropped is not None
        or control_name is not None,
    }
    return {
        "passed": all(checks.values()),
        "control_name": control_name,
        "waived_checks": [],
        "checks": checks,
        "counts": {
            "clock_total": int(len(clock)),
            "train": int(len(windows["train"])),
            "selection": int(len(windows["selection"])),
            **{name: int(len(value)) for name, value in periods.items()},
            **{name: int(len(value)) for name, value in quarters.items()},
        },
        "side_counts": side_counts,
        "year_side_counts": year_side,
        "selection_half_side_counts": half_side,
        "maximum_month_share": {
            name: _max_month_share(window) for name, window in windows.items()
        },
        "delayed_entry_dropped_split_edges": delayed_dropped,
    }


def control_support_summaries(
    controls: pd.DataFrame, packet_audit: dict[str, Any]
) -> dict[str, Any]:
    return {
        name: support_gate_summary(
            controls[controls["clock"] == name],
            packet_audit,
            control_name=name,
        )
        for name in CONTROL_NAMES
    }


def _format_clock(clock: pd.DataFrame) -> pd.DataFrame:
    out = clock.copy()
    for column in (
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


def _frame_hash(clock: pd.DataFrame) -> str:
    return canonical_hash(_format_clock(clock).to_dict(orient="records"))


def _publish_new(temporary: Path, final: Path) -> None:
    os.link(temporary, final)


def build_support_artifacts(cfg: Config) -> dict[str, Any]:
    registration = validate_frozen_preregistration(cfg.preregistration)
    _validate_config(cfg, registration)
    source = load_source_frame(registration)
    exact_source = _exact_source_audit(source)
    if not exact_source["passed"]:
        raise RuntimeError("FETD exact frozen source audit failed")
    packets, packet_audit = build_packets(source)
    features = build_features(packets)
    primary = build_clock(features, mode="primary", clock="primary")
    controls, delayed_dropped = build_control_clocks(features, primary)
    gates = support_gate_summary(
        primary, packet_audit, delayed_dropped=delayed_dropped
    )
    control_gates = control_support_summaries(controls, packet_audit)

    output = _repository_path(cfg.output)
    output_tmp = _temporary_path(output)
    try:
        source_binding = _source_binding(registration)
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
                "path": source_binding["path"],
                "sha256": source_binding["sha256"],
                "bytes": source_binding["bytes"],
                "rows_read": int(len(source)),
                "columns": SOURCE_COLUMNS,
                "signal_columns": sorted(SIGNAL_SOURCE_COLUMNS),
                "forbidden_primary_columns": sorted(
                    FORBIDDEN_SIGNAL_SOURCE_COLUMNS
                ),
                "exact_source_audit": exact_source,
            },
            "packet_audit": packet_audit,
            "feature_audit": {
                "packet_rows": int(len(packets)),
                "base_valid_feature_rows": int(features["feature_valid"].sum()),
                "rank_ready_rows": int(features["rank_ready"].sum()),
                "lookback_valid_feature_packets": 180,
                "minimum_prior_valid_feature_packets": 120,
                "primary_thresholds": {
                    "opposite_transport_signs": True,
                    "strain_rank_minimum": 0.75,
                },
                "source_values_summarized": False,
            },
            "sealed_clock_commitments": {
                "primary": {
                    "frame_hash": _frame_hash(primary),
                    "rows": int(len(primary)),
                    "columns": CLOCK_COLUMNS,
                },
                "controls": {
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
            "event_rows_published": 0,
            "feature_values_published": 0,
            "outcome_boundary": dict(OUTCOME_BOUNDARY),
            "performance_values_opened": False,
            "next_action": (
                "commit and hash-freeze strict evaluator before outcomes"
                if gates["passed"]
                else "reject FETD-288 without opening outcomes"
            ),
            "stopping_rule": (
                "reject permanently without outcomes on any primary support "
                "failure; no threshold, side, rank-window, packet, support-floor, "
                "hold, latency, calendar, or clock repair"
            ),
        }
        artifact = {**core, "manifest_hash": canonical_hash(core)}
        output_tmp.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        _publish_new(output_tmp, output)
        return artifact
    finally:
        output_tmp.unlink(missing_ok=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--output", default=Config.output)
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
