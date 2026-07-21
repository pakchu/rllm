"""Evaluate frozen BLSR-288 source support and novelty without outcomes.

The evaluator opens the hash-bound confirmed-ledger snapshot and, only after
primary/control support passes, the preregistered comparator clocks.  It never
opens BTC market bars, funding, returns, PnL, equity, or post-2023 source data.
"""

# Pandas stubs expose scalar indexing as broad Series/DataFrame/NaT unions.
# Runtime schema validation and synthetic tests narrow every such boundary.
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# pyright: reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportOperatorIssue=false
# pyright: reportReturnType=false

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
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from training import (
    build_fee_endpoint_topology_disagreement_support as ledger,
)
from training import (
    preregister_blockspace_load_settlement_relay as prereg,
)


POLICY_ID = "BLSR-288"
PROTOCOL_VERSION = "blockspace_load_settlement_relay_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SOURCE = Path("training/evaluate_blockspace_load_settlement_relay_support.py")
PREREGISTRATION_ARTIFACT = prereg.DEFAULT_OUTPUT
PREREGISTRATION_ARTIFACT_SHA256 = (
    "d8a79e22a5670e88baa2a0108d79ce515c5f6f9e093615cba9290ec4b1369f5b"
)
PREREGISTRATION_MANIFEST_HASH = (
    "87d4486be4c37d6f6078409c719f35ff196583c80fa834cf0fdf47d5f437a29a"
)
PREREGISTRATION_POLICY_HASH = (
    "f9194cb23d01e910861f573a735293faa087f315c422000ef4da049409f6f62f"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "f26015b4b18af4c6e3ca12330abffb389b71300158c206409eb383215d4f0810"
)
LEDGER_BUILDER = Path("training/build_fee_endpoint_topology_disagreement_support.py")
LEDGER_BUILDER_SHA256 = (
    "1d1330415d6b22f0ebe32719dd6b5232cb4df28b08c2f0ee2942f15aa7c6f01d"
)
FETD_SUPPORT_MANIFEST_HASH = (
    "24902cbe9869d2c5dc3443047d31c7ad0a1650d23822da6158d7e4b5ee758c27"
)
FETD_PRIMARY_FRAME_HASH = (
    "70cb3a0600d611793db4fd6786d9061007328d2768680b534f50ac07de0ebe38"
)
FETD_PRIMARY_ROWS = 119

DEFAULT_OUTPUT_REPORT = Path(
    "results/blockspace_load_settlement_relay_support_2026-07-21.json"
)
ARTIFACT_ROOT = Path("results")

FIVE_MINUTES = pd.Timedelta(minutes=5)
HOLD = pd.Timedelta(hours=24)
TOLERANT_MATCH = pd.Timedelta(hours=6)
EVALUATION_START = pd.Timestamp("2021-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2023-01-01T00:00:00Z")
EVALUATION_END = pd.Timestamp("2024-01-01T00:00:00Z")

FEATURE_COLUMNS = (
    "packet_id",
    "packet_start_height",
    "packet_end_height",
    "confirmation_end_height",
    "source_available_at_utc",
    "feature_valid",
    "rank_ready",
    "fee_change",
    "endpoint_change",
    "fee_magnitude_rank",
    "endpoint_magnitude_rank",
)
CLOCK_COLUMNS = (
    "policy_id",
    "clock",
    "window",
    "onset_packet_id",
    "confirmation_packet_id",
    "decision_time_utc",
    "entry_time_utc",
    "exit_time_utc",
    "side",
)
CONTROL_NAMES = (
    "fee_only",
    "endpoint_only",
    "same_packet_agreement",
    "reverse_order_relay",
    "opposite_response_relay",
    "one_packet_stale_response",
    "direction_flip",
    "deterministic_random_side",
    "one_bar_latency",
)
SUPPORT_LIMITS = {
    "train_total_minimum": 80,
    "train_each_year_minimum": 30,
    "train_each_half_year_minimum": 12,
    "train_each_side_minimum": 24,
    "train_each_side_each_year_minimum": 8,
    "train_maximum_month_share": 0.20,
    "train_maximum_weekday_share": 0.25,
    "selection_total_minimum": 35,
    "selection_each_half_minimum": 14,
    "selection_each_quarter_minimum": 6,
    "selection_each_side_minimum": 12,
    "selection_each_side_each_half_minimum": 4,
    "selection_maximum_month_share": 0.20,
    "selection_maximum_weekday_share": 0.25,
}
NOVELTY_LIMITS = {
    "exact_entry_timestamp_jaccard_maximum": 0.20,
    "candidate_one_to_one_within_six_hours_fraction_maximum": 0.35,
    "signed_occupied_exposure_absolute_pearson_maximum": 0.40,
}
BLSR_SOURCE_COLUMNS = (
    "height",
    "id",
    "previousblockhash",
    "timestamp",
    "weight",
    "total_fees",
    "total_inputs",
    "total_outputs",
)
BLSR_INTEGER_COLUMNS = (
    "height",
    "timestamp",
    "weight",
    "total_fees",
    "total_inputs",
    "total_outputs",
)
FORBIDDEN_SOURCE_VALUE_COLUMNS = (
    "mediantime",
    "tx_count",
    "size",
    "utxo_set_change",
)
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_COMPARATOR_CAPABILITIES = {
    "BATE-288": "directional_interval",
    "CVTR-1": "directional_interval",
    "DFFB-601": "directional_interval",
    "FETD-288": "directional_interval",
    "FLCC-1:FLCC-H4-Q60": "directional_interval",
    "FLCC-1:FLCC-H4-Q65": "directional_interval",
    "FLCC-1:FLCC-H8-Q60": "directional_interval",
    "FLCC-1:FLCC-H8-Q65": "directional_interval",
    "NTB-7": "directional_interval",
    "NWE-7": "timestamp_only",
    "NWE-8": "directional_interval",
    "ORFR-1": "directional_interval",
    "UFCP-1": "directional_interval",
    "WCTR-288": "directional_interval",
    "chain_activity_impulse_momentum": "directional_interval",
    "live_anchor_2023": "timestamp_only",
    "prior_microstructure:cbfr72": "timestamp_only",
    "prior_microstructure:mfic_fast": "timestamp_only",
    "prior_microstructure:mfic_slow": "timestamp_only",
    "prior_microstructure:mfic_union": "timestamp_only",
    "prior_microstructure:netf_fast": "timestamp_only",
    "prior_microstructure:netf_slow": "timestamp_only",
    "prior_microstructure:netf_union": "timestamp_only",
    "prior_microstructure:terminal_absorption_wait72_h72": "timestamp_only",
    "prior_microstructure:wfrs_l288_q90_h144": "timestamp_only",
}


@dataclass(frozen=True)
class Config:
    output_report: str = str(DEFAULT_OUTPUT_REPORT)


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_bound_file(path: str | Path, expected_sha: str, label: str) -> Path:
    target = _repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"BLSR {label} is missing or symlinked")
    if sha256_file(target) != expected_sha:
        raise RuntimeError(f"BLSR {label} SHA drift")
    return target


def load_preregistration() -> dict[str, Any]:
    _require_bound_file(
        PREREGISTRATION_ARTIFACT,
        PREREGISTRATION_ARTIFACT_SHA256,
        "preregistration artifact",
    )
    _require_bound_file(
        prereg.PREREGISTRATION_SOURCE,
        PREREGISTRATION_SOURCE_SHA256,
        "preregistration source",
    )
    _require_bound_file(LEDGER_BUILDER, LEDGER_BUILDER_SHA256, "ledger builder")
    artifact = prereg.load_preregistration()
    if artifact.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("BLSR preregistration manifest drift")
    if artifact.get("policy_hash") != PREREGISTRATION_POLICY_HASH:
        raise RuntimeError("BLSR preregistration policy drift")
    if artifact.get("policy_id") != POLICY_ID:
        raise RuntimeError("BLSR preregistration identity drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("BLSR preregistration opened outcomes")
    if artifact.get("comparator_bindings") != prereg.COMPARATOR_BINDINGS:
        raise RuntimeError("BLSR comparator binding drift")
    if artifact["policy"]["controls"] != prereg.CONTROL_DEFINITIONS:
        raise RuntimeError("BLSR control definitions drift")
    return artifact


def load_source_frame(registration: Mapping[str, Any]) -> pd.DataFrame:
    source_manifest = registration.get("source_manifest")
    if not isinstance(source_manifest, dict):
        raise RuntimeError("BLSR source binding missing")
    source = source_manifest.get("source_output")
    if not isinstance(source, dict):
        raise RuntimeError("BLSR source output binding missing")
    path = _require_bound_file(source["path"], source["sha256"], "ledger source")
    compressed = path.read_bytes()
    if len(compressed) != source.get("bytes"):
        raise RuntimeError("BLSR ledger source byte-size drift")
    frame = pd.read_csv(
        io.BytesIO(compressed),
        compression="gzip",
        usecols=list(BLSR_SOURCE_COLUMNS),
        dtype={"id": "string", "previousblockhash": "string"},
    )
    if tuple(frame.columns) != BLSR_SOURCE_COLUMNS:
        raise RuntimeError("BLSR allowed source-column order drift")
    return validate_source_frame(frame)


def validate_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != BLSR_SOURCE_COLUMNS:
        raise RuntimeError("BLSR source frame contains non-signal columns")
    if frame.empty:
        raise RuntimeError("BLSR source frame is empty")
    out = frame.copy()
    for column in BLSR_INTEGER_COLUMNS:
        values = pd.to_numeric(out[column], errors="raise")
        if values.isna().any() or (values % 1).ne(0).any():
            raise RuntimeError(f"BLSR source {column} must contain exact integers")
        out[column] = values.astype(np.int64)
    out["id"] = out["id"].astype(str)
    out["previousblockhash"] = out["previousblockhash"].astype(str)
    out = out.reset_index(drop=True)

    heights = out["height"].to_numpy(np.int64)
    expected = np.arange(heights[0], heights[0] + len(out), dtype=np.int64)
    if not np.array_equal(heights, expected):
        raise RuntimeError("BLSR source rows must have exact contiguous heights")
    if out["id"].duplicated().any():
        raise RuntimeError("BLSR source block ids must be unique")
    if not out["id"].map(lambda value: bool(HASH_RE.fullmatch(value))).all():
        raise RuntimeError("BLSR source block ids must be lowercase 64-hex")
    if (
        not out["previousblockhash"]
        .map(lambda value: bool(HASH_RE.fullmatch(value)))
        .all()
    ):
        raise RuntimeError("BLSR source previous hashes must be lowercase 64-hex")
    if not np.array_equal(
        out["previousblockhash"].iloc[1:].to_numpy(),
        out["id"].iloc[:-1].to_numpy(),
    ):
        raise RuntimeError("BLSR source hash-chain linkage failed")
    if out[["timestamp", "weight"]].le(0).any().any():
        raise RuntimeError("BLSR source contains non-positive required fields")
    if out[["total_fees", "total_inputs", "total_outputs"]].lt(0).any().any():
        raise RuntimeError("BLSR source contains negative fee/endpoint counts")
    return out


def _exact_source_audit(source: pd.DataFrame) -> dict[str, Any]:
    checks = {
        "allowed_columns_only": tuple(source.columns) == BLSR_SOURCE_COLUMNS,
        "forbidden_source_values_not_loaded": not set(
            FORBIDDEN_SOURCE_VALUE_COLUMNS
        ).intersection(source.columns),
        "exact_rows": len(source) == prereg.source_contract.FROZEN_ROWS,
        "exact_start_height": int(source["height"].iloc[0])
        == prereg.source_contract.FROZEN_START_HEIGHT,
        "exact_end_height": int(source["height"].iloc[-1])
        == prereg.source_contract.FROZEN_END_HEIGHT,
        "all_rows_pre_cutoff": bool(
            source["timestamp"]
            .lt(prereg.source_contract.FROZEN_END_TIMESTAMP_EXCLUSIVE)
            .all()
        ),
        "unique_block_hashes": not source["id"].duplicated().any(),
        "hash_chain_contiguous": bool(
            np.array_equal(
                source["previousblockhash"].iloc[1:].to_numpy(),
                source["id"].iloc[:-1].to_numpy(),
            )
        ),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def build_packets(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = validate_source_frame(source).copy()
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
            raise RuntimeError("BLSR complete packet rows are not contiguous")
        confirmation_index = end_index + 6
        if confirmation_index >= len(frame):
            raise RuntimeError("BLSR complete packet lacks six successors")
        confirmation_end_height = int(frame.at[confirmation_index, "height"])
        if confirmation_end_height != expected_start + 77:
            raise RuntimeError("BLSR packet confirmation height drift")
        availability_timestamp = (
            int(frame.loc[start_index:confirmation_index, "timestamp"].max()) + 172_800
        )
        available = pd.Timestamp(availability_timestamp, unit="s", tz="UTC")
        entry = available.ceil("5min") + FIVE_MINUTES
        total_weight = int(group["weight"].sum())
        total_fees = int(group["total_fees"].sum())
        total_endpoints = int((group["total_inputs"] + group["total_outputs"]).sum())
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
                "exit_time_utc": entry + HOLD,
            }
        )
    packets = pd.DataFrame.from_records(rows)
    if packets.empty:
        raise RuntimeError("BLSR source produced no complete packets")
    packets = packets.sort_values("packet_id", kind="stable").reset_index(drop=True)
    packet_ids = packets["packet_id"].to_numpy(np.int64)
    expected_ids = np.arange(packet_ids[0], packet_ids[0] + len(packet_ids))
    if not np.array_equal(packet_ids, expected_ids):
        raise RuntimeError("BLSR complete packet ids are not consecutive")
    allowed_partial = {
        int(frame["packet_id"].iloc[0]),
        int(frame["packet_id"].iloc[-1]),
    }
    if not set(partial_packet_ids).issubset(allowed_partial):
        raise RuntimeError("BLSR source contains an incomplete interior packet")
    return packets, {
        "complete_packets": int(len(packets)),
        "first_complete_packet_id": int(packet_ids[0]),
        "last_complete_packet_id": int(packet_ids[-1]),
        "first_complete_packet_start_height": int(
            packets["packet_start_height"].iloc[0]
        ),
        "last_complete_packet_end_height": int(packets["packet_end_height"].iloc[-1]),
        "partial_edge_packet_ids": partial_packet_ids,
        "all_complete_packets_have_72_blocks": True,
        "complete_packet_ids_consecutive": True,
        "all_confirmation_blocks_contained": True,
    }


def _strict_prior_midrank(current: float, prior: Iterable[float]) -> float:
    values = list(prior)
    if not values:
        raise ValueError("BLSR strict-prior midrank requires prior values")
    less = sum(value < current for value in values)
    equal = sum(value == current for value in values)
    return (less + 0.5 * equal) / len(values)


def _utc_series(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    if parsed.isna().any():
        raise RuntimeError(f"BLSR {label} contains missing timestamps")
    return parsed


def build_features(
    packets: pd.DataFrame,
    *,
    lookback: int = 180,
    minimum_prior: int = 120,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive one-packet changes and strict-prior magnitude ranks."""
    if lookback < minimum_prior or minimum_prior <= 0:
        raise ValueError("BLSR rank lookback/minimum contract is invalid")
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
        raise RuntimeError("BLSR packet frame schema drift")
    ordered = packets.sort_values("packet_id", kind="stable").reset_index(drop=True)
    if ordered.empty or ordered["packet_id"].duplicated().any():
        raise RuntimeError("BLSR packet ids must be nonempty and unique")
    if not ordered["packet_id"].diff().dropna().eq(1).all():
        raise RuntimeError("BLSR packet ids must be consecutive")
    available = _utc_series(ordered["source_available_at_utc"], "packet clock")
    availability_delta = available.diff().dropna()
    availability_strict = bool((availability_delta > pd.Timedelta(0)).all())
    if not availability_strict:
        raise RuntimeError("BLSR packet availability is not strictly increasing")
    ordered["source_available_at_utc"] = available

    records: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for index, row in ordered.iterrows():
        fee_change = math.nan
        endpoint_change = math.nan
        fee_rank = math.nan
        endpoint_rank = math.nan
        feature_valid = False
        rank_ready = False
        if index >= 1:
            prior = ordered.iloc[index - 1]
            if bool(row["packet_valid"]) and bool(prior["packet_valid"]):
                fee_change = float(row["fee_pressure"]) - float(prior["fee_pressure"])
                endpoint_change = float(row["endpoint_density"]) - float(
                    prior["endpoint_density"]
                )
                feature_valid = bool(
                    math.isfinite(fee_change) and math.isfinite(endpoint_change)
                )
        current_available = cast(pd.Timestamp, row["source_available_at_utc"])
        if feature_valid:
            causal_prior = [
                item for item in history if item["available_at"] < current_available
            ][-lookback:]
            if len(causal_prior) >= minimum_prior:
                fee_rank = _strict_prior_midrank(
                    abs(fee_change),
                    (item["fee_magnitude"] for item in causal_prior),
                )
                endpoint_rank = _strict_prior_midrank(
                    abs(endpoint_change),
                    (item["endpoint_magnitude"] for item in causal_prior),
                )
                rank_ready = bool(
                    math.isfinite(fee_rank) and math.isfinite(endpoint_rank)
                )
            history.append(
                {
                    "available_at": current_available,
                    "fee_magnitude": abs(fee_change),
                    "endpoint_magnitude": abs(endpoint_change),
                }
            )
        records.append(
            {
                "packet_id": int(row["packet_id"]),
                "packet_start_height": int(row["packet_start_height"]),
                "packet_end_height": int(row["packet_end_height"]),
                "confirmation_end_height": int(row["confirmation_end_height"]),
                "source_available_at_utc": current_available,
                "feature_valid": feature_valid,
                "rank_ready": rank_ready,
                "fee_change": fee_change,
                "endpoint_change": endpoint_change,
                "fee_magnitude_rank": fee_rank,
                "endpoint_magnitude_rank": endpoint_rank,
            }
        )
    features = pd.DataFrame.from_records(records, columns=FEATURE_COLUMNS)
    return features, {
        "availability_strictly_increasing": availability_strict,
        "feature_rows": int(len(features)),
        "base_valid_feature_rows": int(features["feature_valid"].sum()),
        "rank_ready_rows": int(features["rank_ready"].sum()),
        "lookback_valid_packet_changes": lookback,
        "minimum_prior_valid_packet_changes": minimum_prior,
    }


def _is_significant(row: Any, family: str) -> bool:
    if not bool(row.rank_ready):
        return False
    if family == "fee":
        value = float(row.fee_change)
        rank = float(row.fee_magnitude_rank)
    elif family == "endpoint":
        value = float(row.endpoint_change)
        rank = float(row.endpoint_magnitude_rank)
    else:
        raise ValueError(f"unknown BLSR feature family {family!r}")
    return bool(value != 0.0 and math.isfinite(value) and rank >= 0.75)


def _feature_sign(row: Any, family: str) -> int:
    value = float(row.fee_change if family == "fee" else row.endpoint_change)
    if value == 0.0 or not math.isfinite(value):
        raise RuntimeError("BLSR significant feature has no finite sign")
    return 1 if value > 0.0 else -1


def _entry_after(available: Any) -> pd.Timestamp:
    return pd.Timestamp(available).ceil("5min") + FIVE_MINUTES


def _candidate_record(
    *,
    clock: str,
    onset_packet_id: int,
    confirmation_packet_id: int,
    decision_time: Any,
    side: int,
) -> dict[str, Any]:
    decision = pd.Timestamp(decision_time)
    entry = _entry_after(decision)
    return {
        "clock": clock,
        "onset_packet_id": int(onset_packet_id),
        "confirmation_packet_id": int(confirmation_packet_id),
        "decision_time_utc": decision,
        "entry_time_utc": entry,
        "exit_time_utc": entry + HOLD,
        "side": int(side),
    }


def relay_candidates(
    features: pd.DataFrame,
    *,
    clock: str,
    onset_family: str,
    response_family: str,
    confirm_same_sign: bool,
    stale_response_packets: int = 0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build a fixed three-packet first-response relay before scheduling."""
    if stale_response_packets not in (0, 1):
        raise ValueError("BLSR only freezes zero or one stale response packet")
    if tuple(features.columns) != FEATURE_COLUMNS:
        raise RuntimeError("BLSR feature frame schema drift")
    ordered = features.sort_values("packet_id", kind="stable").reset_index(drop=True)
    records: list[dict[str, Any]] = []
    active: dict[str, int] | None = None
    audit = {
        "onsets": 0,
        "same_sign_resolutions": 0,
        "opposite_sign_resolutions": 0,
        "emitted_candidates": 0,
        "expired": 0,
        "open_at_source_end": 0,
        "active_fee_or_endpoint_shocks_ignored": 0,
    }
    for index, row in enumerate(ordered.itertuples(index=False)):
        packet_id = int(row.packet_id)
        if active is not None:
            delta = packet_id - active["packet_id"]
            if delta < 1 or delta > 3:
                raise RuntimeError("BLSR relay packet deadline drift")
            if _is_significant(row, onset_family):
                audit["active_fee_or_endpoint_shocks_ignored"] += 1
            response_index = index - stale_response_packets
            response = ordered.iloc[response_index] if response_index >= 0 else None
            response_is_later = bool(
                response is not None
                and int(response["packet_id"]) > active["packet_id"]
            )
            response_significant = bool(
                response_is_later and _is_significant(response, response_family)
            )
            if response_significant:
                response_sign = _feature_sign(response, response_family)
                same = response_sign == active["side"]
                key = "same_sign_resolutions" if same else "opposite_sign_resolutions"
                audit[key] += 1
                if same is confirm_same_sign:
                    records.append(
                        _candidate_record(
                            clock=clock,
                            onset_packet_id=active["packet_id"],
                            confirmation_packet_id=packet_id,
                            decision_time=row.source_available_at_utc,
                            side=active["side"],
                        )
                    )
                    audit["emitted_candidates"] += 1
                active = None
                continue
            if delta == 3:
                audit["expired"] += 1
                active = None
                continue
            continue
        if _is_significant(row, onset_family):
            active = {
                "packet_id": packet_id,
                "side": _feature_sign(row, onset_family),
            }
            audit["onsets"] += 1
    if active is not None:
        audit["open_at_source_end"] = 1
    columns = [
        column for column in CLOCK_COLUMNS if column not in ("policy_id", "window")
    ]
    return pd.DataFrame.from_records(records, columns=columns), audit


def _single_packet_candidates(
    features: pd.DataFrame, *, clock: str, mode: str
) -> pd.DataFrame:
    if mode not in {"fee_only", "endpoint_only", "same_packet_agreement"}:
        raise ValueError(f"unknown BLSR single-packet control {mode!r}")
    records: list[dict[str, Any]] = []
    for row in features.sort_values("packet_id", kind="stable").itertuples(index=False):
        fee = _is_significant(row, "fee")
        endpoint = _is_significant(row, "endpoint")
        side = 0
        if mode == "fee_only" and fee:
            side = _feature_sign(row, "fee")
        elif mode == "endpoint_only" and endpoint:
            side = _feature_sign(row, "endpoint")
        elif mode == "same_packet_agreement" and fee and endpoint:
            fee_side = _feature_sign(row, "fee")
            if fee_side == _feature_sign(row, "endpoint"):
                side = fee_side
        if side == 0:
            continue
        records.append(
            _candidate_record(
                clock=clock,
                onset_packet_id=int(row.packet_id),
                confirmation_packet_id=int(row.packet_id),
                decision_time=row.source_available_at_utc,
                side=side,
            )
        )
    columns = [
        column for column in CLOCK_COLUMNS if column not in ("policy_id", "window")
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def _window_for(entry: pd.Timestamp, exit_time: pd.Timestamp) -> str | None:
    if EVALUATION_START <= entry and exit_time <= TRAIN_END:
        return "train"
    if TRAIN_END <= entry and exit_time <= EVALUATION_END:
        return "selection"
    return None


def schedule_candidates(
    candidates: pd.DataFrame, *, clock: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    raw_columns = [
        column for column in CLOCK_COLUMNS if column not in ("policy_id", "window")
    ]
    if tuple(candidates.columns) != tuple(raw_columns):
        raise RuntimeError("BLSR candidate frame schema drift")
    ordered = candidates.sort_values(
        ["entry_time_utc", "onset_packet_id", "confirmation_packet_id"],
        kind="stable",
    )
    accepted: list[dict[str, Any]] = []
    prior_exit: pd.Timestamp | None = None
    dropped = {"split_containment": 0, "global_overlap": 0}
    for row in ordered.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time_utc)
        exit_time = pd.Timestamp(row.exit_time_utc)
        window = _window_for(entry, exit_time)
        if window is None:
            dropped["split_containment"] += 1
            continue
        if prior_exit is not None and entry < prior_exit:
            dropped["global_overlap"] += 1
            continue
        accepted.append(
            {
                "policy_id": POLICY_ID,
                "clock": clock,
                "window": window,
                "onset_packet_id": int(row.onset_packet_id),
                "confirmation_packet_id": int(row.confirmation_packet_id),
                "decision_time_utc": pd.Timestamp(row.decision_time_utc),
                "entry_time_utc": entry,
                "exit_time_utc": exit_time,
                "side": int(row.side),
            }
        )
        prior_exit = exit_time
    return pd.DataFrame.from_records(accepted, columns=CLOCK_COLUMNS), dropped


def _same_clock(
    primary: pd.DataFrame, *, clock: str, sides: Iterable[int]
) -> pd.DataFrame:
    out = primary.copy()
    out["clock"] = clock
    out["side"] = list(sides)
    return out.loc[:, list(CLOCK_COLUMNS)]


def _random_side(entry: Any) -> int:
    timestamp = pd.Timestamp(entry).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = hashlib.sha256(
        f"BLSR-288-random-side-20260721|{timestamp}".encode("ascii")
    ).digest()
    return 1 if digest[0] < 128 else -1


def _latency_control(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    delayed = primary.copy()
    delayed["clock"] = "one_bar_latency"
    delayed["entry_time_utc"] += FIVE_MINUTES
    delayed["exit_time_utc"] += FIVE_MINUTES
    contained = pd.Series(
        [
            _window_for(
                pd.Timestamp(row.entry_time_utc), pd.Timestamp(row.exit_time_utc)
            )
            == row.window
            for row in delayed.itertuples(index=False)
        ],
        index=delayed.index,
        dtype=bool,
    )
    dropped = {
        "train": int(((~contained) & delayed["window"].eq("train")).sum()),
        "selection": int(((~contained) & delayed["window"].eq("selection")).sum()),
    }
    return delayed.loc[contained, list(CLOCK_COLUMNS)].reset_index(drop=True), dropped


def build_clocks(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    primary_candidates, primary_relay = relay_candidates(
        features,
        clock="primary",
        onset_family="fee",
        response_family="endpoint",
        confirm_same_sign=True,
    )
    primary, primary_drops = schedule_candidates(primary_candidates, clock="primary")

    relay_controls: dict[str, tuple[pd.DataFrame, dict[str, int]]] = {}
    for name, onset, response, same, stale in (
        ("reverse_order_relay", "endpoint", "fee", True, 0),
        ("opposite_response_relay", "fee", "endpoint", False, 0),
        ("one_packet_stale_response", "fee", "endpoint", True, 1),
    ):
        candidates, relay_audit = relay_candidates(
            features,
            clock=name,
            onset_family=onset,
            response_family=response,
            confirm_same_sign=same,
            stale_response_packets=stale,
        )
        clock, drops = schedule_candidates(candidates, clock=name)
        relay_controls[name] = (clock, {**relay_audit, **drops})

    direct_controls: dict[str, tuple[pd.DataFrame, dict[str, int]]] = {}
    for name in ("fee_only", "endpoint_only", "same_packet_agreement"):
        candidates = _single_packet_candidates(features, clock=name, mode=name)
        clock, drops = schedule_candidates(candidates, clock=name)
        direct_controls[name] = (clock, drops)

    direction = _same_clock(
        primary, clock="direction_flip", sides=-primary["side"].astype(int)
    )
    random_side = _same_clock(
        primary,
        clock="deterministic_random_side",
        sides=(_random_side(value) for value in primary["entry_time_utc"]),
    )
    latency, latency_drops = _latency_control(primary)
    controls_by_name = {
        **{name: value[0] for name, value in direct_controls.items()},
        **{name: value[0] for name, value in relay_controls.items()},
        "direction_flip": direction,
        "deterministic_random_side": random_side,
        "one_bar_latency": latency,
    }
    nonempty_controls = [
        controls_by_name[name]
        for name in CONTROL_NAMES
        if not controls_by_name[name].empty
    ]
    controls = (
        pd.concat(nonempty_controls, ignore_index=True)
        if nonempty_controls
        else pd.DataFrame(columns=CLOCK_COLUMNS)
    )
    clocks = (
        pd.concat([primary, controls], ignore_index=True)
        .sort_values(
            ["clock", "entry_time_utc", "onset_packet_id", "confirmation_packet_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return clocks, {
        "primary_candidates": int(len(primary_candidates)),
        "primary_relay": primary_relay,
        "primary_drops": primary_drops,
        "control_candidate_and_drop_audit": {
            **{name: audit for name, (_, audit) in direct_controls.items()},
            **{name: audit for name, (_, audit) in relay_controls.items()},
            "direction_flip": {},
            "deterministic_random_side": {},
            "one_bar_latency": latency_drops,
        },
    }


def _period_count(frame: pd.DataFrame, start: str, end: str) -> int:
    entry = frame["entry_time_utc"]
    return int(((entry >= pd.Timestamp(start)) & (entry < pd.Timestamp(end))).sum())


def _maximum_share(frame: pd.DataFrame, period: str) -> float:
    if frame.empty:
        return 1.0
    entry = frame["entry_time_utc"]
    if period == "month":
        bucket = entry.dt.strftime("%Y-%m")
    elif period == "weekday":
        bucket = entry.dt.weekday
    else:
        raise ValueError(f"unsupported BLSR concentration period {period!r}")
    return float(bucket.value_counts(normalize=True).max())


def support_summary(
    primary: pd.DataFrame,
    packet_audit: Mapping[str, Any],
    feature_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if tuple(primary.columns) != CLOCK_COLUMNS:
        raise RuntimeError("BLSR primary clock schema drift")
    train = primary.loc[primary["window"].eq("train")]
    selection = primary.loc[primary["window"].eq("selection")]
    periods = {
        name: _period_count(primary, start, end)
        for name, start, end in (
            ("2021", "2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
            ("2022", "2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
            ("2021H1", "2021-01-01T00:00:00Z", "2021-07-01T00:00:00Z"),
            ("2021H2", "2021-07-01T00:00:00Z", "2022-01-01T00:00:00Z"),
            ("2022H1", "2022-01-01T00:00:00Z", "2022-07-01T00:00:00Z"),
            ("2022H2", "2022-07-01T00:00:00Z", "2023-01-01T00:00:00Z"),
            ("2023H1", "2023-01-01T00:00:00Z", "2023-07-01T00:00:00Z"),
            ("2023H2", "2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"),
            ("2023Q1", "2023-01-01T00:00:00Z", "2023-04-01T00:00:00Z"),
            ("2023Q2", "2023-04-01T00:00:00Z", "2023-07-01T00:00:00Z"),
            ("2023Q3", "2023-07-01T00:00:00Z", "2023-10-01T00:00:00Z"),
            ("2023Q4", "2023-10-01T00:00:00Z", "2024-01-01T00:00:00Z"),
        )
    }
    side_counts = {
        "train": {
            "long": int(train["side"].eq(1).sum()),
            "short": int(train["side"].eq(-1).sum()),
        },
        "selection": {
            "long": int(selection["side"].eq(1).sum()),
            "short": int(selection["side"].eq(-1).sum()),
        },
    }
    train_year_side = {
        year: {
            side_name: int(
                (
                    primary["entry_time_utc"].ge(
                        pd.Timestamp(f"{year}-01-01T00:00:00Z")
                    )
                    & primary["entry_time_utc"].lt(
                        pd.Timestamp(f"{int(year) + 1}-01-01T00:00:00Z")
                    )
                    & primary["side"].eq(side)
                ).sum()
            )
            for side_name, side in (("long", 1), ("short", -1))
        }
        for year in ("2021", "2022")
    }
    selection_half_side = {
        half: {
            side_name: int(
                (
                    primary["entry_time_utc"].ge(pd.Timestamp(start))
                    & primary["entry_time_utc"].lt(pd.Timestamp(end))
                    & primary["side"].eq(side)
                ).sum()
            )
            for side_name, side in (("long", 1), ("short", -1))
        }
        for half, start, end in (
            ("2023H1", "2023-01-01T00:00:00Z", "2023-07-01T00:00:00Z"),
            ("2023H2", "2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"),
        )
    }
    concentration = {
        name: {
            "maximum_month_share": _maximum_share(frame, "month"),
            "maximum_weekday_share": _maximum_share(frame, "weekday"),
        }
        for name, frame in (("train", train), ("selection", selection))
    }
    source_checks = {
        "complete_packet_count": packet_audit.get("complete_packets") == 2_959,
        "first_complete_packet_start": packet_audit.get(
            "first_complete_packet_start_height"
        )
        == 610_704,
        "last_complete_packet_end": packet_audit.get("last_complete_packet_end_height")
        == 823_751,
        "complete_packets_have_72_blocks": packet_audit.get(
            "all_complete_packets_have_72_blocks"
        )
        is True,
        "complete_packet_ids_consecutive": packet_audit.get(
            "complete_packet_ids_consecutive"
        )
        is True,
        "confirmation_blocks_contained": packet_audit.get(
            "all_confirmation_blocks_contained"
        )
        is True,
        "availability_strictly_increasing": feature_audit.get(
            "availability_strictly_increasing"
        )
        is True,
    }
    support_checks = {
        "train_total_minimum": len(train) >= SUPPORT_LIMITS["train_total_minimum"],
        "train_each_year_minimum": all(
            periods[year] >= SUPPORT_LIMITS["train_each_year_minimum"]
            for year in ("2021", "2022")
        ),
        "train_each_half_year_minimum": all(
            periods[half] >= SUPPORT_LIMITS["train_each_half_year_minimum"]
            for half in ("2021H1", "2021H2", "2022H1", "2022H2")
        ),
        "train_each_side_minimum": min(side_counts["train"].values())
        >= SUPPORT_LIMITS["train_each_side_minimum"],
        "train_each_side_each_year_minimum": all(
            train_year_side[year][side]
            >= SUPPORT_LIMITS["train_each_side_each_year_minimum"]
            for year in ("2021", "2022")
            for side in ("long", "short")
        ),
        "train_maximum_month_share": concentration["train"]["maximum_month_share"]
        <= SUPPORT_LIMITS["train_maximum_month_share"],
        "train_maximum_weekday_share": concentration["train"]["maximum_weekday_share"]
        <= SUPPORT_LIMITS["train_maximum_weekday_share"],
        "selection_total_minimum": len(selection)
        >= SUPPORT_LIMITS["selection_total_minimum"],
        "selection_each_half_minimum": all(
            periods[half] >= SUPPORT_LIMITS["selection_each_half_minimum"]
            for half in ("2023H1", "2023H2")
        ),
        "selection_each_quarter_minimum": all(
            periods[quarter] >= SUPPORT_LIMITS["selection_each_quarter_minimum"]
            for quarter in ("2023Q1", "2023Q2", "2023Q3", "2023Q4")
        ),
        "selection_each_side_minimum": min(side_counts["selection"].values())
        >= SUPPORT_LIMITS["selection_each_side_minimum"],
        "selection_each_side_each_half_minimum": all(
            selection_half_side[half][side]
            >= SUPPORT_LIMITS["selection_each_side_each_half_minimum"]
            for half in ("2023H1", "2023H2")
            for side in ("long", "short")
        ),
        "selection_maximum_month_share": concentration["selection"][
            "maximum_month_share"
        ]
        <= SUPPORT_LIMITS["selection_maximum_month_share"],
        "selection_maximum_weekday_share": concentration["selection"][
            "maximum_weekday_share"
        ]
        <= SUPPORT_LIMITS["selection_maximum_weekday_share"],
    }
    checks = {**source_checks, **support_checks}
    return {
        "passed": bool(all(checks.values())),
        "limits": dict(SUPPORT_LIMITS),
        "checks": checks,
        "counts": {
            "total": int(len(primary)),
            "train": int(len(train)),
            "selection": int(len(selection)),
            **periods,
        },
        "side_counts": side_counts,
        "train_year_side_counts": train_year_side,
        "selection_half_side_counts": selection_half_side,
        "concentration": concentration,
    }


def control_structure_summary(clocks: pd.DataFrame) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name in CONTROL_NAMES:
        frame = clocks.loc[clocks["clock"].eq(name)].sort_values(
            ["entry_time_utc", "onset_packet_id", "confirmation_packet_id"],
            kind="stable",
        )
        expected_window = [
            _window_for(
                pd.Timestamp(row.entry_time_utc), pd.Timestamp(row.exit_time_utc)
            )
            for row in frame.itertuples(index=False)
        ]
        previous_exit = frame["exit_time_utc"].shift(1)
        checks = {
            "clock_identity": bool(frame["clock"].eq(name).all()),
            "decision_before_entry": bool(
                (frame["decision_time_utc"] <= frame["entry_time_utc"]).all()
            ),
            "exact_24h_hold": bool(
                (frame["exit_time_utc"] - frame["entry_time_utc"]).eq(HOLD).all()
            ),
            "valid_side": bool(frame["side"].isin((-1, 1)).all()),
            "split_contained": expected_window == frame["window"].tolist(),
            "global_nonoverlap": bool(
                (
                    frame["entry_time_utc"].iloc[1:].array
                    >= previous_exit.iloc[1:].array
                ).all()
            ),
            "post_2023_absent": bool(
                frame["entry_time_utc"].lt(EVALUATION_END).all()
                and frame["exit_time_utc"].le(EVALUATION_END).all()
            ),
            "unique_incidence": not frame.duplicated(
                ["entry_time_utc", "onset_packet_id", "confirmation_packet_id"]
            ).any(),
        }
        summaries[name] = {
            "passed": bool(all(checks.values())),
            "rows": int(len(frame)),
            "checks": checks,
        }
    return {
        "passed": bool(all(value["passed"] for value in summaries.values())),
        "controls": summaries,
    }


COMPARATOR_COLUMNS = (
    "comparator",
    "capability",
    "entry_time",
    "exit_time",
    "side",
    "source_clock",
)


def _directional_comparator(
    frame: pd.DataFrame,
    *,
    comparator: str,
    entry_column: str,
    exit_column: str,
    side_column: str,
    source_clock: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "comparator": comparator,
            "capability": "directional_interval",
            "entry_time": _utc_series(frame[entry_column], comparator),
            "exit_time": _utc_series(frame[exit_column], comparator),
            "side": pd.to_numeric(frame[side_column], errors="raise").astype(np.int8),
            "source_clock": source_clock,
        }
    )
    return out.loc[:, list(COMPARATOR_COLUMNS)]


def _load_fetd_comparator(packets: pd.DataFrame) -> pd.DataFrame:
    support_binding = prereg.COMPARATOR_BINDINGS["fetd_288_support"]
    support_path = _require_bound_file(
        support_binding["path"], support_binding["sha256"], "FETD support artifact"
    )
    support = json.loads(support_path.read_text(encoding="utf-8"))
    if support.get("manifest_hash") != FETD_SUPPORT_MANIFEST_HASH:
        raise RuntimeError("BLSR FETD support manifest drift")
    commitment = support.get("sealed_clock_commitments", {}).get("primary", {})
    if commitment.get("frame_hash") != FETD_PRIMARY_FRAME_HASH:
        raise RuntimeError("BLSR FETD primary commitment drift")
    if commitment.get("rows") != FETD_PRIMARY_ROWS:
        raise RuntimeError("BLSR FETD primary row-count drift")
    features = ledger.build_features(packets)
    primary = ledger.build_clock(features, mode="primary", clock="primary")
    if ledger._frame_hash(primary) != FETD_PRIMARY_FRAME_HASH:
        raise RuntimeError("BLSR FETD rebuilt clock hash drift")
    return _directional_comparator(
        primary,
        comparator="FETD-288",
        entry_column="entry_time_utc",
        exit_column="exit_time_utc",
        side_column="side",
        source_clock="FETD-288:primary",
    )


def _load_simple_comparators() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []

    bate_binding = prereg.COMPARATOR_BINDINGS["bate_288_primary_clock"]
    bate_path = _require_bound_file(
        bate_binding["path"], bate_binding["sha256"], "BATE primary clock"
    )
    bate = pd.read_csv(
        bate_path, usecols=["policy_id", "entry_time", "exit_time", "side"]
    )
    if not bate["policy_id"].eq("BATE-288").all():
        raise RuntimeError("BLSR BATE identity drift")
    frames.append(
        _directional_comparator(
            bate,
            comparator="BATE-288",
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
            source_clock="BATE-288:primary",
        )
    )

    ufcp_binding = prereg.COMPARATOR_BINDINGS["ufcp_1_primary_clock"]
    ufcp_path = _require_bound_file(
        ufcp_binding["path"], ufcp_binding["sha256"], "UFCP primary clock"
    )
    ufcp = pd.read_csv(
        ufcp_path,
        usecols=["policy_id", "clock", "entry_time", "exit_time", "side"],
    )
    if (
        not ufcp["policy_id"].eq("UFCP-1").all()
        or not ufcp["clock"].eq("primary").all()
    ):
        raise RuntimeError("BLSR UFCP identity drift")
    frames.append(
        _directional_comparator(
            ufcp,
            comparator="UFCP-1",
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
            source_clock="UFCP-1:primary",
        )
    )

    wctr_binding = prereg.COMPARATOR_BINDINGS["wctr_288_primary_clock"]
    wctr_path = _require_bound_file(
        wctr_binding["path"], wctr_binding["sha256"], "WCTR primary clock"
    )
    wctr = pd.read_csv(
        wctr_path,
        usecols=["policy_id", "clock", "entry_time_utc", "exit_time_utc", "side"],
    )
    if (
        not wctr["policy_id"].eq("WCTR-288").all()
        or not wctr["clock"].eq("primary").all()
    ):
        raise RuntimeError("BLSR WCTR identity drift")
    frames.append(
        _directional_comparator(
            wctr,
            comparator="WCTR-288",
            entry_column="entry_time_utc",
            exit_column="exit_time_utc",
            side_column="side",
            source_clock="WCTR-288:primary",
        )
    )
    return frames


def _load_prior_bundle() -> pd.DataFrame:
    clock_binding = prereg.COMPARATOR_BINDINGS["prior_microstructure_bundle"]
    manifest_binding = prereg.COMPARATOR_BINDINGS[
        "prior_microstructure_bundle_manifest"
    ]
    clock_path = _require_bound_file(
        clock_binding["path"], clock_binding["sha256"], "prior comparator bundle"
    )
    manifest_path = _require_bound_file(
        manifest_binding["path"],
        manifest_binding["sha256"],
        "prior comparator bundle manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clock_meta = manifest.get("clock", {})
    expected_columns = [
        "comparator",
        "capability",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
        "source_clock",
    ]
    if clock_meta.get("columns") != expected_columns:
        raise RuntimeError("BLSR prior comparator header commitment drift")
    frame = pd.read_csv(clock_path)
    if frame.columns.tolist() != expected_columns:
        raise RuntimeError("BLSR prior comparator header drift")
    if len(frame) != clock_meta.get("rows"):
        raise RuntimeError("BLSR prior comparator row-count drift")
    observed_counts = {
        str(name): int(count)
        for name, count in frame["comparator"].value_counts().sort_index().items()
    }
    if observed_counts != clock_meta.get("counts"):
        raise RuntimeError("BLSR prior comparator identity counts drift")
    decision = _utc_series(frame["decision_time"], "prior comparator decision")
    entry = _utc_series(frame["entry_time"], "prior comparator entry")
    if not bool((decision <= entry).all()):
        raise RuntimeError("BLSR prior comparator decides after entry")
    out = frame.loc[
        :,
        ["comparator", "capability", "entry_time", "exit_time", "side", "source_clock"],
    ].copy()
    out["entry_time"] = entry
    return out.loc[:, list(COMPARATOR_COLUMNS)]


def load_comparators(packets: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = [
        _load_fetd_comparator(packets),
        *_load_simple_comparators(),
        _load_prior_bundle(),
    ]
    combined = pd.concat(frames, ignore_index=True)
    _validate_comparator_identities(combined)
    identities = sorted(EXPECTED_COMPARATOR_CAPABILITIES)
    return combined.loc[:, list(COMPARATOR_COLUMNS)], {
        "rows_read": int(len(combined)),
        "identities": identities,
        "identity_count": len(identities),
        "directional_rows": int(
            combined["capability"].eq("directional_interval").sum()
        ),
        "timestamp_only_rows": int(combined["capability"].eq("timestamp_only").sum()),
    }


def _validate_comparator_identities(comparators: pd.DataFrame) -> None:
    if tuple(comparators.columns) != COMPARATOR_COLUMNS:
        raise RuntimeError("BLSR comparator frame schema drift")
    observed: dict[str, str] = {}
    for name, frame in comparators.groupby("comparator", sort=True, observed=True):
        capabilities = set(str(value) for value in frame["capability"].unique())
        if len(capabilities) != 1:
            raise RuntimeError(f"BLSR comparator capability drift: {name}")
        observed[str(name)] = next(iter(capabilities))
    if observed != EXPECTED_COMPARATOR_CAPABILITIES:
        raise RuntimeError("BLSR exact comparator identity/capability set drift")


def exact_entry_jaccard(candidate: pd.Series, comparator: pd.Series) -> dict[str, Any]:
    left = set(_utc_series(candidate, "candidate entry").astype("int64"))
    right = set(_utc_series(comparator, "comparator entry").astype("int64"))
    union = left | right
    intersection = left & right
    return {
        "candidate_unique_entries": len(left),
        "comparator_unique_entries": len(right),
        "intersection_entries": len(intersection),
        "exact_entry_timestamp_jaccard": (
            float(len(intersection) / len(union)) if union else 1.0
        ),
    }


def one_to_one_tolerant_overlap(
    candidate: pd.Series,
    comparator: pd.Series,
    *,
    tolerance: pd.Timedelta = TOLERANT_MATCH,
) -> dict[str, Any]:
    left = np.sort(_utc_series(candidate, "candidate tolerant entry").astype("int64"))
    right = np.sort(
        _utc_series(comparator, "comparator tolerant entry").astype("int64")
    )
    tolerance_ns = int(tolerance.value)
    left_index = 0
    right_index = 0
    matches = 0
    while left_index < len(left) and right_index < len(right):
        delta = int(right[right_index]) - int(left[left_index])
        if delta < -tolerance_ns:
            right_index += 1
        elif delta > tolerance_ns:
            left_index += 1
        else:
            matches += 1
            left_index += 1
            right_index += 1
    return {
        "tolerance_seconds": int(tolerance.total_seconds()),
        "candidate_rows": int(len(left)),
        "comparator_rows": int(len(right)),
        "one_to_one_matches": matches,
        "candidate_one_to_one_within_six_hours_fraction": (
            float(matches / len(left)) if len(left) else 1.0
        ),
    }


def _validated_intervals(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    checked = frame.loc[:, ["entry_time", "exit_time", "side"]].copy()
    checked["entry_time"] = _utc_series(checked["entry_time"], f"{label} entry")
    checked["exit_time"] = _utc_series(checked["exit_time"], f"{label} exit")
    checked["side"] = pd.to_numeric(checked["side"], errors="raise").astype(np.int8)
    if not checked["side"].isin((-1, 1)).all():
        raise RuntimeError(f"BLSR {label} has invalid side")
    if not (checked["entry_time"] < checked["exit_time"]).all():
        raise RuntimeError(f"BLSR {label} has invalid interval")
    step_ns = int(FIVE_MINUTES.value)
    if bool((checked["entry_time"].astype("int64") % step_ns).ne(0).any()) or bool(
        (checked["exit_time"].astype("int64") % step_ns).ne(0).any()
    ):
        raise RuntimeError(f"BLSR {label} intervals are not 5m aligned")
    checked = checked.sort_values("entry_time", kind="stable").reset_index(drop=True)
    if len(checked) > 1 and bool(
        (
            checked["entry_time"].iloc[1:].array
            < checked["exit_time"].shift(1).iloc[1:].array
        ).any()
    ):
        raise RuntimeError(f"BLSR {label} intervals overlap")
    return checked


def signed_occupied_exposure_correlation(
    candidate: pd.DataFrame, comparator: pd.DataFrame
) -> dict[str, Any]:
    left = candidate.rename(
        columns={"entry_time_utc": "entry_time", "exit_time_utc": "exit_time"}
    )
    left = _validated_intervals(left, "candidate")
    right = _validated_intervals(comparator, "comparator")
    periods = int((EVALUATION_END - EVALUATION_START) // FIVE_MINUTES)

    def exposure(frame: pd.DataFrame) -> np.ndarray:
        values = np.zeros(periods, dtype=np.int8)
        for row in frame.itertuples(index=False):
            start = max(pd.Timestamp(row.entry_time), EVALUATION_START)
            end = min(pd.Timestamp(row.exit_time), EVALUATION_END)
            if start >= end:
                continue
            first = int((start - EVALUATION_START) // FIVE_MINUTES)
            last = int((end - EVALUATION_START) // FIVE_MINUTES)
            if bool(np.any(values[first:last] != 0)):
                raise RuntimeError("BLSR occupied exposure is not flat/long/short")
            values[first:last] = int(row.side)
        return values

    left_values = exposure(left)
    right_values = exposure(right)
    left_variance = float(np.var(left_values.astype(np.float64)))
    right_variance = float(np.var(right_values.astype(np.float64)))
    if left_variance == 0.0 or right_variance == 0.0:
        return {
            "defined": False,
            "failure_reason": "zero_variance",
            "absolute_signed_occupied_exposure_pearson": None,
            "candidate_nonflat_rows": int(np.count_nonzero(left_values)),
            "comparator_nonflat_rows": int(np.count_nonzero(right_values)),
        }
    correlation = float(
        np.corrcoef(left_values.astype(np.float64), right_values.astype(np.float64))[
            0, 1
        ]
    )
    if not math.isfinite(correlation):
        raise RuntimeError("BLSR occupied exposure correlation is non-finite")
    return {
        "defined": True,
        "failure_reason": None,
        "signed_occupied_exposure_pearson": correlation,
        "absolute_signed_occupied_exposure_pearson": abs(correlation),
        "candidate_nonflat_rows": int(np.count_nonzero(left_values)),
        "comparator_nonflat_rows": int(np.count_nonzero(right_values)),
        "grid_rows_5m": periods,
    }


def novelty_summary(primary: pd.DataFrame, comparators: pd.DataFrame) -> dict[str, Any]:
    if tuple(comparators.columns) != COMPARATOR_COLUMNS:
        raise RuntimeError("BLSR comparator frame schema drift")
    candidate = primary.loc[
        primary["entry_time_utc"].ge(EVALUATION_START)
        & primary["entry_time_utc"].lt(EVALUATION_END)
    ].copy()
    if candidate["entry_time_utc"].duplicated().any():
        raise RuntimeError("BLSR candidate entry times are not unique")
    summaries: dict[str, Any] = {}
    for name in sorted(str(value) for value in comparators["comparator"].unique()):
        reference = comparators.loc[comparators["comparator"].eq(name)].copy()
        capabilities = set(str(value) for value in reference["capability"].unique())
        if len(capabilities) != 1:
            raise RuntimeError(f"BLSR comparator capability drift: {name}")
        capability = next(iter(capabilities))
        reference["entry_time"] = _utc_series(reference["entry_time"], name)
        relevant = reference.loc[
            reference["entry_time"].ge(EVALUATION_START)
            & reference["entry_time"].lt(EVALUATION_END)
        ].copy()
        if relevant.empty:
            raise RuntimeError(f"BLSR comparator has no evaluation entries: {name}")
        step_ns = int(FIVE_MINUTES.value)
        if bool((relevant["entry_time"].astype("int64") % step_ns).ne(0).any()):
            raise RuntimeError(f"BLSR comparator entry is not 5m aligned: {name}")
        exact = exact_entry_jaccard(candidate["entry_time_utc"], relevant["entry_time"])
        tolerant = one_to_one_tolerant_overlap(
            candidate["entry_time_utc"], relevant["entry_time"]
        )
        checks = {
            "exact_entry_timestamp_jaccard": exact["exact_entry_timestamp_jaccard"]
            <= NOVELTY_LIMITS["exact_entry_timestamp_jaccard_maximum"],
            "candidate_one_to_one_within_six_hours_fraction": tolerant[
                "candidate_one_to_one_within_six_hours_fraction"
            ]
            <= NOVELTY_LIMITS["candidate_one_to_one_within_six_hours_fraction_maximum"],
        }
        exposure: dict[str, Any] | None = None
        interval_overlap_rows: int | None = None
        if capability == "directional_interval":
            if reference["exit_time"].isna().any() or reference["side"].isna().any():
                raise RuntimeError(f"BLSR directional comparator incomplete: {name}")
            reference["exit_time"] = _utc_series(reference["exit_time"], f"{name} exit")
            interval_relevant = reference.loc[
                reference["entry_time"].lt(EVALUATION_END)
                & reference["exit_time"].gt(EVALUATION_START)
            ].copy()
            interval_overlap_rows = int(len(interval_relevant))
            exposure = signed_occupied_exposure_correlation(
                candidate, interval_relevant
            )
            checks["signed_occupied_exposure_defined"] = bool(exposure["defined"])
            checks["signed_occupied_exposure_pearson"] = bool(
                exposure["defined"]
                and exposure["absolute_signed_occupied_exposure_pearson"]
                <= NOVELTY_LIMITS["signed_occupied_exposure_absolute_pearson_maximum"]
            )
        elif capability == "timestamp_only":
            if (
                not reference["exit_time"].isna().all()
                or not reference["side"].isna().all()
            ):
                raise RuntimeError(
                    f"BLSR timestamp-only comparator leaks direction: {name}"
                )
        else:
            raise RuntimeError(f"BLSR unknown comparator capability: {name}")
        summaries[name] = {
            "passed": bool(all(checks.values())),
            "capability": capability,
            "rows": int(len(relevant)),
            "interval_overlap_rows": interval_overlap_rows,
            "checks": checks,
            "exact_entry_overlap": exact,
            "one_to_one_tolerant_overlap": tolerant,
            "signed_occupied_exposure": exposure,
        }
    return {
        "evaluated": True,
        "passed": bool(
            summaries and all(value["passed"] for value in summaries.values())
        ),
        "limits": dict(NOVELTY_LIMITS),
        "comparators": summaries,
    }


def _format_clock(clock: pd.DataFrame) -> pd.DataFrame:
    out = clock.copy()
    for column in ("decision_time_utc", "entry_time_utc", "exit_time_utc"):
        out[column] = _utc_series(out[column], "clock commitment").dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    return out.loc[:, list(CLOCK_COLUMNS)]


def _frame_hash(clock: pd.DataFrame) -> str:
    return canonical_hash(_format_clock(clock).to_dict(orient="records"))


def evaluate_source_only(
    source: pd.DataFrame,
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    exact_source = _exact_source_audit(source)
    if not exact_source["passed"]:
        raise RuntimeError("BLSR exact frozen source audit failed")
    packets, packet_audit = build_packets(source)
    features, feature_audit = build_features(packets)
    clocks, relay_audit = build_clocks(features)
    primary = clocks.loc[clocks["clock"].eq("primary")].reset_index(drop=True)
    controls = control_structure_summary(clocks)
    support = support_summary(primary, packet_audit, feature_audit)

    comparators_read = 0
    comparator_audit: dict[str, Any] | None = None
    if support["passed"] and controls["passed"]:
        comparators, comparator_audit = load_comparators(packets)
        comparators_read = int(len(comparators))
        novelty = novelty_summary(primary, comparators)
    else:
        novelty = {
            "evaluated": False,
            "passed": False,
            "skip_reason": "primary support or control structure failed",
            "limits": dict(NOVELTY_LIMITS),
            "comparators": {},
        }
    passed = bool(support["passed"] and controls["passed"] and novelty["passed"])
    failed_stages = [
        name
        for name, result in (
            ("support", support["passed"]),
            ("control_structure", controls["passed"]),
            ("novelty", novelty["passed"]),
        )
        if not result
    ]
    control_frame = clocks.loc[clocks["clock"].ne("primary")].reset_index(drop=True)
    source_binding = registration["source_manifest"]["source_output"]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": POLICY_ID,
        "source_only": True,
        "market_outcomes_opened": False,
        "performance_values_opened": False,
        "source": {
            "path": source_binding["path"],
            "sha256": source_binding["sha256"],
            "rows_read": int(len(source)),
            "signal_columns": list(BLSR_SOURCE_COLUMNS),
            "forbidden_source_value_columns": list(FORBIDDEN_SOURCE_VALUE_COLUMNS),
            "exact_source_audit": exact_source,
        },
        "packet_audit": packet_audit,
        "feature_audit": {
            **feature_audit,
            "source_values_summarized": False,
        },
        "relay_audit": relay_audit,
        "support": support,
        "control_structure": controls,
        "novelty": novelty,
        "comparator_audit": comparator_audit,
        "sealed_clock_commitments": {
            "primary": {
                "frame_hash": _frame_hash(primary),
                "rows": int(len(primary)),
                "columns": list(CLOCK_COLUMNS),
            },
            "controls": {
                "frame_hash": _frame_hash(control_frame),
                "rows": int(len(control_frame)),
                "columns": list(CLOCK_COLUMNS),
                "clock_counts": {
                    name: int(control_frame["clock"].eq(name).sum())
                    for name in CONTROL_NAMES
                },
            },
        },
        "event_rows_published": 0,
        "feature_values_published": 0,
        "verdict": {
            "passed": passed,
            "status": "PASS" if passed else "REJECT",
            "failed_stages": failed_stages,
            "strict_economic_train_authorized": passed,
            "repair_allowed_under_candidate_identity": False,
        },
        "outcome_boundary": {
            "source_value_rows_read": int(len(source)),
            "source_feature_rows_derived": int(len(features)),
            "primary_event_incidence_rows_derived": int(len(primary)),
            "control_event_incidence_rows_derived": int(len(control_frame)),
            "comparator_event_rows_read": comparators_read,
            "btc_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "return_rows_loaded": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_source_rows_read": 0,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
        "next_action": (
            "commit and hash-freeze strict economic train evaluator"
            if passed
            else "retire BLSR-288 without outcomes or identity repair"
        ),
    }


def _protected_paths(registration: Mapping[str, Any]) -> set[Path]:
    protected = {
        _repository_path(EVALUATOR_SOURCE),
        _repository_path(PREREGISTRATION_ARTIFACT),
        _repository_path(prereg.PREREGISTRATION_SOURCE),
        _repository_path(prereg.SOURCE_MANIFEST),
        _repository_path(prereg.SOURCE_VALIDATOR),
        _repository_path(prereg.MECHANISM_DECISION),
        _repository_path(LEDGER_BUILDER),
        _repository_path(registration["source_manifest"]["source_output"]["path"]),
    }
    protected.update(
        _repository_path(binding["path"])
        for binding in prereg.COMPARATOR_BINDINGS.values()
    )
    return protected


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_evaluation(cfg: Config | None = None) -> dict[str, Any]:
    frozen_cfg = Config() if cfg is None else cfg
    registration = load_preregistration()
    output = _repository_path(frozen_cfg.output_report)
    if output.suffix != ".json":
        raise ValueError("BLSR support report must be JSON")
    if not output.is_relative_to(_repository_path(ARTIFACT_ROOT)):
        raise ValueError("BLSR support report must stay under results")
    if output in _protected_paths(registration):
        raise ValueError("BLSR support report aliases a protected input")
    if output.exists():
        raise FileExistsError("BLSR support report is immutable")
    source = load_source_frame(registration)
    core = evaluate_source_only(source, registration)
    core.update(
        {
            "config": asdict(frozen_cfg),
            "evaluator_source": {
                "path": str(EVALUATOR_SOURCE),
                "sha256": sha256_file(EVALUATOR_SOURCE),
            },
            "preregistration": {
                "path": str(PREREGISTRATION_ARTIFACT),
                "sha256": PREREGISTRATION_ARTIFACT_SHA256,
                "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
                "policy_hash": PREREGISTRATION_POLICY_HASH,
            },
            "ledger_builder": {
                "path": str(LEDGER_BUILDER),
                "sha256": LEDGER_BUILDER_SHA256,
            },
        }
    )
    report = {**core, "manifest_hash": canonical_hash(core)}
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_new(output, payload)
    return report


def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-report", default=str(DEFAULT_OUTPUT_REPORT))
    args = parser.parse_args(argv)
    return Config(output_report=args.output_report)


def main(argv: Sequence[str] | None = None) -> int:
    report = run_evaluation(parse_args(argv))
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "status": report["verdict"]["status"],
                "primary_rows": report["sealed_clock_commitments"]["primary"]["rows"],
                "novelty_evaluated": report["novelty"]["evaluated"],
                "manifest_hash": report["manifest_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
