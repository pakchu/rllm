"""Build the outcome-blind BATE-288 clock from Bitcoin block summaries only."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import download_bitcoin_block_summaries as block_source


PREREGISTRATION = (
    "docs/block-arrival-throughput-elasticity-bate288-support-"
    "preregistration-2026-07-20.md"
)
PREREGISTRATION_SHA256 = (
    "bdb8a9b7e602ca671fb496045a59741a5dd77fea7ed11a66c374e5fd6b5893cd"
)
DEFAULT_SOURCE = "data/bitcoin_block_summaries_2020_2023.csv.gz"
DEFAULT_SOURCE_MANIFEST = (
    "results/bitcoin_block_summaries_source_manifest_2026-07-20.json"
)
DEFAULT_OUTPUT = (
    "results/block_arrival_throughput_elasticity_support_2026-07-20.json"
)
DEFAULT_CLOCK = (
    "results/block_arrival_throughput_elasticity_clock_2026-07-20.csv"
)
PROTOCOL_VERSION = "block_arrival_throughput_elasticity_support_v1"
FROZEN_SOURCE_LOADER_SHA256 = (
    "0628e9e5925087e68b5d0a7a8f74dc91d908a209d140557a8ac690cd0e98bc53"
)
TRAIN_START = pd.Timestamp("2021-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2023-01-01T00:00:00Z")
SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")
CLOCK_COLUMNS = (
    "policy_id",
    "side",
    "state",
    "packet_end_height",
    "confirmation_height",
    "entry_time",
    "exit_time",
    "elapsed_seconds",
    "weight_log",
    "tx_log",
    "weight_z",
    "tx_z",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "BATE-288"
    packet_blocks: int = 6
    reference_packets: int = 2_016
    mad_consistency_scale: float = 1.4826
    z_threshold: float = 1.25
    confirmation_blocks: int = 6
    historical_embargo_seconds: int = 7_200
    bar_seconds: int = 300
    entry_delay_bars: int = 1
    hold_bars: int = 288
    minimum_positive_span_ratio: float = 0.995
    maximum_invalid_span_run: int = 12


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    return block_source.sha256_file(path)


def _validate_policy(policy: Policy) -> None:
    if policy != Policy():
        raise RuntimeError("BATE-288 support policy differs from preregistration")


def _manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_hash"}


def _validate_frozen_loader(path: str | Path) -> None:
    if sha256_file(path) != FROZEN_SOURCE_LOADER_SHA256:
        raise RuntimeError("BATE source loader differs from the frozen SHA-256")


def load_source(source_csv: str, source_manifest: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    prereg = Path(PREREGISTRATION)
    if not prereg.is_file() or sha256_file(prereg) != PREREGISTRATION_SHA256:
        raise RuntimeError("BATE-288 support preregistration is missing or has drifted")
    _validate_frozen_loader(Path(block_source.__file__))
    manifest_path = Path(source_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("Bitcoin block source manifest is not an object")
    if manifest.get("protocol_version") != block_source.PROTOCOL_VERSION:
        raise RuntimeError("Bitcoin block source protocol differs from BATE freeze")
    expected_hash = canonical_hash(_manifest_core(manifest))
    if manifest.get("manifest_hash") != expected_hash:
        raise RuntimeError("Bitcoin block source manifest hash mismatch")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Bitcoin block source manifest has no config")
    frozen = {
        "start_height": block_source.FROZEN_START_HEIGHT,
        "end_height": block_source.FROZEN_END_HEIGHT,
        "end_timestamp_exclusive": block_source.FIRST_2024_TIMESTAMP,
    }
    if any(config.get(key) != value for key, value in frozen.items()):
        raise RuntimeError("Bitcoin block source is not the frozen full prefix")
    decision = manifest.get("source_decision")
    if not isinstance(decision, dict) or decision.get(
        "sha256"
    ) != block_source.SOURCE_DECISION_SHA256:
        raise RuntimeError("Bitcoin block source decision hash mismatch")
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise RuntimeError("Bitcoin block source manifest has no output contract")
    if output.get("columns") != list(block_source.OUTPUT_COLUMNS):
        raise RuntimeError("Bitcoin block source columns differ from BATE freeze")
    if sha256_file(source_csv) != output.get("sha256"):
        raise RuntimeError("Bitcoin block source file hash mismatch")
    boundary = manifest.get("outcome_boundary")
    expected_boundary = {
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "return_or_pnl_fields": 0,
        "post_2023_source_rows_loaded": 0,
        "raw_esplora_responses_persisted": False,
    }
    if boundary != expected_boundary:
        raise RuntimeError("Bitcoin block source crossed the outcome boundary")

    frame = pd.read_csv(
        source_csv,
        usecols=list(block_source.OUTPUT_COLUMNS),
        dtype={"id": "string", "previousblockhash": "string"},
    )
    if len(frame) != block_source.FROZEN_END_HEIGHT - block_source.FROZEN_START_HEIGHT + 1:
        raise RuntimeError("Bitcoin block source row count differs from BATE freeze")
    for column in (
        "height",
        "timestamp",
        "mediantime",
        "tx_count",
        "size",
        "weight",
    ):
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or not np.issubdtype(values.dtype, np.integer):
            raise RuntimeError(f"Bitcoin block source {column} must be exact integers")
        frame[column] = values.astype(np.int64)
    frame = frame.sort_values("height").reset_index(drop=True)
    expected_heights = np.arange(
        block_source.FROZEN_START_HEIGHT,
        block_source.FROZEN_END_HEIGHT + 1,
        dtype=np.int64,
    )
    if not np.array_equal(frame["height"].to_numpy(), expected_heights):
        raise RuntimeError("Bitcoin block source is not the exact inclusive range")
    ids = frame["id"].astype(str)
    previous = frame["previousblockhash"].astype(str)
    if ids.duplicated().any() or not np.array_equal(
        previous.iloc[1:].to_numpy(), ids.iloc[:-1].to_numpy()
    ):
        raise RuntimeError("Bitcoin block source hash-chain linkage failed")
    if frame["timestamp"].ge(block_source.FIRST_2024_TIMESTAMP).any():
        raise RuntimeError("Bitcoin block source crossed the sealed 2024 boundary")
    if (
        frame[["timestamp", "mediantime", "tx_count", "size", "weight"]]
        .le(0)
        .any()
        .any()
    ):
        raise RuntimeError("Bitcoin block source contains non-positive fields")
    if frame["weight"].gt(block_source.MAX_BLOCK_WEIGHT).any() or (
        frame["weight"].lt(frame["size"])
        | frame["weight"].gt(4 * frame["size"])
    ).any():
        raise RuntimeError("Bitcoin block source size/weight invariant failed")
    return frame, manifest


def _strict_prior_robust_z(
    values: np.ndarray,
    *,
    reference: int,
    consistency_scale: float,
    batch_size: int = 1_024,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite_indexes = np.flatnonzero(np.isfinite(values))
    finite_values = values[finite_indexes]
    if len(finite_values) <= reference:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(
        finite_values, reference + 1
    )
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        chunk = windows[start:stop]
        prior = chunk[:, :-1]
        center = np.median(prior, axis=1)
        mad = np.median(np.abs(prior - center[:, None]), axis=1)
        scale = consistency_scale * mad
        valid_scale = np.isfinite(scale) & (scale > 0.0)
        z = np.full(len(chunk), np.nan, dtype=np.float64)
        z[valid_scale] = (
            chunk[valid_scale, -1] - center[valid_scale]
        ) / scale[valid_scale]
        target = finite_indexes[reference + start : reference + stop]
        result[target] = z
    return result


def _maximum_true_run(mask: np.ndarray) -> int:
    maximum = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def build_features(blocks: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    required = set(block_source.OUTPUT_COLUMNS)
    if set(blocks.columns) != required:
        raise RuntimeError("BATE feature input must contain block-only frozen columns")
    ordered = blocks.sort_values("height").reset_index(drop=True)
    heights = ordered["height"].to_numpy(np.int64)
    if len(ordered) < 2 * policy.packet_blocks + 1:
        raise ValueError("BATE feature input is too short for packet and confirmations")
    if not np.all(np.diff(heights) == 1):
        raise RuntimeError("BATE feature input heights must be contiguous")
    timestamps = ordered["timestamp"].to_numpy(np.int64)
    weights = ordered["weight"].to_numpy(np.float64)
    tx_counts = ordered["tx_count"].to_numpy(np.float64)
    packet_index = np.arange(
        policy.packet_blocks,
        len(ordered) - policy.confirmation_blocks,
        dtype=np.int64,
    )
    elapsed = timestamps[packet_index] - timestamps[
        packet_index - policy.packet_blocks
    ]
    weight_cumulative = np.concatenate(([0.0], np.cumsum(weights)))
    tx_cumulative = np.concatenate(([0.0], np.cumsum(tx_counts)))
    weight_sum = weight_cumulative[packet_index + 1] - weight_cumulative[
        packet_index - policy.packet_blocks + 1
    ]
    tx_sum = tx_cumulative[packet_index + 1] - tx_cumulative[
        packet_index - policy.packet_blocks + 1
    ]
    raw_valid = elapsed > 0
    weight_log = np.full(len(packet_index), np.nan, dtype=np.float64)
    tx_log = np.full(len(packet_index), np.nan, dtype=np.float64)
    weight_log[raw_valid] = np.log(weight_sum[raw_valid] / elapsed[raw_valid])
    tx_log[raw_valid] = np.log(tx_sum[raw_valid] / elapsed[raw_valid])
    weight_z = _strict_prior_robust_z(
        weight_log,
        reference=policy.reference_packets,
        consistency_scale=policy.mad_consistency_scale,
    )
    tx_z = _strict_prior_robust_z(
        tx_log,
        reference=policy.reference_packets,
        consistency_scale=policy.mad_consistency_scale,
    )
    state_valid = np.isfinite(weight_z) & np.isfinite(tx_z)
    state = np.zeros(len(packet_index), dtype=np.int8)
    state[state_valid & (weight_z >= policy.z_threshold) & (tx_z >= policy.z_threshold)] = 1
    state[state_valid & (weight_z <= -policy.z_threshold) & (tx_z <= -policy.z_threshold)] = -1
    onset = _state_onsets(state, state_valid)
    confirmation_height = heights[packet_index + policy.confirmation_blocks]
    span = policy.packet_blocks + policy.confirmation_blocks + 1
    timestamp_windows = np.lib.stride_tricks.sliding_window_view(timestamps, span)
    raw_available = (
        timestamp_windows[packet_index - policy.packet_blocks].max(axis=1)
        + policy.historical_embargo_seconds
    )
    decision_boundary = (
        (raw_available + policy.bar_seconds - 1) // policy.bar_seconds
    ) * policy.bar_seconds
    entry_epoch = decision_boundary + policy.entry_delay_bars * policy.bar_seconds

    return pd.DataFrame(
        {
            "packet_end_height": heights[packet_index],
            "confirmation_height": confirmation_height,
            "elapsed_seconds": elapsed,
            "raw_valid": raw_valid,
            "weight_log": weight_log,
            "tx_log": tx_log,
            "weight_z": weight_z,
            "tx_z": tx_z,
            "state_valid": state_valid,
            "state": state,
            "onset": onset,
            "entry_time": pd.to_datetime(entry_epoch, unit="s", utc=True),
        }
    )


def _state_onsets(state: np.ndarray, state_valid: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.int8)
    state_valid = np.asarray(state_valid, dtype=bool)
    if state.shape != state_valid.shape:
        raise ValueError("BATE state and validity arrays must have equal shape")
    onset = np.zeros(len(state), dtype=bool)
    previous_valid_state: int | None = None
    for index in range(len(state)):
        if not state_valid[index]:
            continue
        current = int(state[index])
        if current in {-1, 1} and current != previous_valid_state:
            onset[index] = True
        previous_valid_state = current
    return onset


def schedule_clock(features: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    candidates = features.loc[features["onset"]].sort_values(
        "packet_end_height"
    )
    accepted: list[int] = []
    next_entry: pd.Timestamp | None = None
    hold = pd.Timedelta(seconds=policy.hold_bars * policy.bar_seconds)
    for index, row in candidates.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        if next_entry is not None and entry < next_entry:
            continue
        accepted.append(index)
        next_entry = entry + hold
    clock = candidates.loc[accepted].copy()
    clock = clock.loc[
        clock["entry_time"].ge(TRAIN_START)
        & clock["entry_time"].lt(SELECTION_END)
    ].reset_index(drop=True)
    clock.insert(0, "policy_id", policy.policy_id)
    clock.insert(1, "side", clock["state"].astype(int))
    clock["state"] = clock["state"].map({1: "HIGH", -1: "LOW"})
    clock["exit_time"] = clock["entry_time"] + hold
    return clock[list(CLOCK_COLUMNS)]


def _window(clock: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    entry = pd.to_datetime(clock["entry_time"], utc=True)
    return clock.loc[
        entry.ge(pd.Timestamp(start, tz="UTC"))
        & entry.lt(pd.Timestamp(end, tz="UTC"))
    ].copy()


def _month_share(clock: pd.DataFrame) -> float:
    if clock.empty:
        return 1.0
    months = (
        pd.to_datetime(clock["entry_time"], utc=True)
        .dt.tz_convert(None)
        .dt.to_period("M")
    )
    return float(months.value_counts().max() / len(clock))


def support_summary(clock: pd.DataFrame) -> dict[str, Any]:
    windows = {
        "train": _window(clock, "2021-01-01", "2023-01-01"),
        "train_2021": _window(clock, "2021-01-01", "2022-01-01"),
        "train_2022": _window(clock, "2022-01-01", "2023-01-01"),
        "train_2021_h1": _window(clock, "2021-01-01", "2021-07-01"),
        "train_2021_h2": _window(clock, "2021-07-01", "2022-01-01"),
        "train_2022_h1": _window(clock, "2022-01-01", "2022-07-01"),
        "train_2022_h2": _window(clock, "2022-07-01", "2023-01-01"),
        "selection": _window(clock, "2023-01-01", "2024-01-01"),
        "selection_h1": _window(clock, "2023-01-01", "2023-07-01"),
        "selection_h2": _window(clock, "2023-07-01", "2024-01-01"),
    }
    counts = {name: int(len(frame)) for name, frame in windows.items()}
    side_counts = {
        name: {
            "HIGH": int((frame["state"] == "HIGH").sum()),
            "LOW": int((frame["state"] == "LOW").sum()),
        }
        for name, frame in windows.items()
    }
    selection_entry = pd.to_datetime(
        windows["selection"]["entry_time"], utc=True
    ).dt.tz_convert(None)
    quarter_counts = {
        str(period): int(count)
        for period, count in selection_entry.dt.to_period("Q").value_counts().sort_index().items()
    }
    checks = {
        "train_total": counts["train"] >= 80,
        "train_each_year": counts["train_2021"] >= 32 and counts["train_2022"] >= 32,
        "train_each_side": side_counts["train"]["HIGH"] >= 25
        and side_counts["train"]["LOW"] >= 25,
        "train_each_side_each_year": all(
            side_counts[year][side] >= 10
            for year in ("train_2021", "train_2022")
            for side in ("HIGH", "LOW")
        ),
        "train_each_half": all(
            counts[half] >= 14
            for half in (
                "train_2021_h1",
                "train_2021_h2",
                "train_2022_h1",
                "train_2022_h2",
            )
        ),
        "train_month_concentration": _month_share(windows["train"]) <= 0.15,
        "selection_total": counts["selection"] >= 35,
        "selection_each_side": side_counts["selection"]["HIGH"] >= 12
        and side_counts["selection"]["LOW"] >= 12,
        "selection_each_half": counts["selection_h1"] >= 14
        and counts["selection_h2"] >= 14,
        "selection_each_side_each_half": all(
            side_counts[half][side] >= 5
            for half in ("selection_h1", "selection_h2")
            for side in ("HIGH", "LOW")
        ),
        "selection_each_quarter": len(quarter_counts) == 4
        and min(quarter_counts.values(), default=0) >= 6,
        "selection_month_concentration": _month_share(windows["selection"]) <= 0.20,
    }
    return {
        "counts": counts,
        "side_counts": side_counts,
        "selection_quarter_counts": quarter_counts,
        "train_maximum_month_share": _month_share(windows["train"]),
        "selection_maximum_month_share": _month_share(windows["selection"]),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def source_support_summary(
    features: pd.DataFrame,
    policy: Policy,
    *,
    maximum_confirmation_height: int = block_source.FROZEN_END_HEIGHT,
) -> dict[str, Any]:
    raw_valid = features["raw_valid"].to_numpy(bool)
    positive_span_ratio = float(raw_valid.mean()) if len(raw_valid) else 0.0
    maximum_invalid_run = _maximum_true_run(~raw_valid)
    confirmation_contained = bool(
        len(features)
        and features["confirmation_height"].max()
        <= maximum_confirmation_height
    )
    checks = {
        "positive_span_ratio": positive_span_ratio
        >= policy.minimum_positive_span_ratio,
        "maximum_invalid_span_run": maximum_invalid_run
        <= policy.maximum_invalid_span_run,
        "confirmation_containment": confirmation_contained,
    }
    return {
        "positive_elapsed_span_ratio": positive_span_ratio,
        "maximum_invalid_elapsed_run": maximum_invalid_run,
        "confirmation_contained": confirmation_contained,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _write_clock(path: str | Path, clock: pd.DataFrame) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        clock.to_csv(
            temporary,
            index=False,
            columns=list(CLOCK_COLUMNS),
            date_format="%Y-%m-%dT%H:%M:%SZ",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
            float_format="%.17g",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def run(
    *,
    source_csv: str = DEFAULT_SOURCE,
    source_manifest: str = DEFAULT_SOURCE_MANIFEST,
    output: str = DEFAULT_OUTPUT,
    clock_output: str = DEFAULT_CLOCK,
) -> dict[str, Any]:
    protected_inputs = {
        Path(source_csv).resolve(),
        Path(source_manifest).resolve(),
        Path(PREREGISTRATION).resolve(),
    }
    artifact_paths = {Path(output).resolve(), Path(clock_output).resolve()}
    if len(artifact_paths) != 2 or artifact_paths & protected_inputs:
        raise ValueError("BATE support artifact paths must be distinct from inputs")
    policy = Policy()
    _validate_policy(policy)
    blocks, manifest = load_source(source_csv, source_manifest)
    features = build_features(blocks, policy)
    source_support = source_support_summary(features, policy)
    positive_span_ratio = source_support["positive_elapsed_span_ratio"]
    maximum_invalid_run = source_support["maximum_invalid_elapsed_run"]
    clock = schedule_clock(features, policy)
    support = support_summary(clock)
    support["source_integrity"] = source_support
    support["checks"] = {
        **source_support["checks"],
        **support["checks"],
    }
    support["passed"] = bool(all(support["checks"].values()))
    _write_clock(clock_output, clock)
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "outcomes_opened": False,
        "policy": asdict(policy),
        "preregistration": {
            "path": PREREGISTRATION,
            "sha256": PREREGISTRATION_SHA256,
        },
        "source": {
            "path": source_csv,
            "sha256": sha256_file(source_csv),
            "manifest_path": source_manifest,
            "manifest_sha256": sha256_file(source_manifest),
            "manifest_hash": manifest["manifest_hash"],
            "frozen_loader_sha256": FROZEN_SOURCE_LOADER_SHA256,
            "rows": int(len(blocks)),
            "columns_loaded": list(block_source.OUTPUT_COLUMNS),
            "market_or_funding_rows_loaded": 0,
            "return_or_pnl_fields_loaded": 0,
            "post_2023_source_rows_loaded": 0,
        },
        "feature_support": {
            "packet_endings": int(len(features)),
            "positive_elapsed_spans": int(features["raw_valid"].sum()),
            "invalid_elapsed_spans": int((~features["raw_valid"]).sum()),
            "positive_elapsed_span_ratio": positive_span_ratio,
            "maximum_invalid_elapsed_run": maximum_invalid_run,
            "finite_state_rows": int(features["state_valid"].sum()),
            "high_state_rows": int((features["state"] == 1).sum()),
            "low_state_rows": int((features["state"] == -1).sum()),
            "onsets_before_nonoverlap": int(features["onset"].sum()),
            "accepted_2021_2023_events": int(len(clock)),
        },
        "clock": {
            "path": clock_output,
            "sha256": sha256_file(clock_output),
            "rows": int(len(clock)),
            "first_entry": str(clock["entry_time"].min()) if len(clock) else None,
            "last_entry": str(clock["entry_time"].max()) if len(clock) else None,
        },
        "support_gate": support,
        "sealed_market_outcomes": ["2021", "2022", "2023", "2024", "2025", "2026_ytd"],
        "failure_action": (
            "freeze the exact source clock and strict evaluator before opening outcomes"
            if support["passed"]
            else "reject BATE-288 without loading any market or funding outcome"
        ),
    }
    result = {**core, "result_hash": canonical_hash(core)}
    _write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", default=DEFAULT_SOURCE)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clock-output", default=DEFAULT_CLOCK)
    args = parser.parse_args()
    result = run(**vars(args))
    print(
        json.dumps(
            {
                "outcomes_opened": result["outcomes_opened"],
                "feature_support": result["feature_support"],
                "support_gate": result["support_gate"],
                "clock": result["clock"],
                "result_hash": result["result_hash"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
