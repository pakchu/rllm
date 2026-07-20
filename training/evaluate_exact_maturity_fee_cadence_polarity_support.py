"""Evaluate outcome-blind source support for the frozen EMFC-864 policy."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right, insort_right
from collections import deque
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_exact_maturity_fee_cadence_polarity as prereg


PROTOCOL_VERSION = "exact_maturity_fee_cadence_polarity_support_v1"
PREREGISTRATION = Path(
    "results/exact_maturity_fee_cadence_polarity_preregistration_2026-07-20.json"
)
PREREGISTRATION_SHA256 = (
    "43f1505786ad5ddd8a076afebccc26bff65387d8ef9b7a443035136606157ff6"
)
PREREGISTRATION_MANIFEST_HASH = (
    "487a4c0dd3aa501605274f0afaacb6714c668078e6fac0506798afa4f9b0d743"
)
PREREGISTRATION_POLICY_HASH = (
    "a264e58f834f2a58dda9ddcf3dcf5035ef941cd2087124b3ea1c8c306559b92f"
)
DEFAULT_SOURCE = Path("data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz")
DEFAULT_SOURCE_MANIFEST = prereg.SOURCE_MANIFEST
DEFAULT_OUTPUT = Path(
    "results/exact_maturity_fee_cadence_polarity_support_2026-07-20.json"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "results/exact_maturity_fee_cadence_polarity_clocks_2026-07-20.csv"
)
SUPPORT_SOURCE = Path(
    "training/evaluate_exact_maturity_fee_cadence_polarity_support.py"
)
GRID_START = pd.Timestamp("2021-01-01T00:00:00Z")
GRID_END = pd.Timestamp("2024-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2023-01-01T00:00:00Z")
BAR_SECONDS = 300
STATE_LABELS = {
    -1: "HIGH_PRESSURE_COMPRESSED",
    1: "LOW_PRESSURE_EXPANDED",
}
CLOCK_COLUMNS = (
    "policy_id",
    "clock",
    "side",
    "state",
    "origin_lag",
    "maturity_height",
    "confirmation_height",
    "entry_time",
    "exit_time",
    "fee_pressure",
    "cadence_compression",
    "fee_rank",
    "cadence_rank",
)
OUTCOME_BOUNDARY = {
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "return_rows_loaded": 0,
    "market_values_read": 0,
    "funding_values_read": 0,
    "return_or_pnl_fields_loaded": 0,
    "post_2023_source_rows_loaded": 0,
}


@dataclass(frozen=True)
class Policy:
    policy_id: str = "EMFC-864"
    maturity_lag: int = 100
    confirmation_blocks: int = 6
    historical_embargo_seconds: int = 7_200
    entry_delay_bars: int = 1
    reference_valid_heights: int = 26_208
    upper_rank: float = 0.90
    lower_rank: float = 0.10
    daily_reference_days: int = 180
    hold_bars: int = 864
    train_total_minimum: int = 60
    train_total_maximum: int = 200
    train_each_year_minimum: int = 24
    selection_total_minimum: int = 24
    selection_total_maximum: int = 105
    selection_each_half_minimum: int = 10
    selection_each_quarter_minimum: int = 3
    side_share_minimum: float = 0.25
    side_share_maximum: float = 0.75
    each_side_each_train_year_minimum: int = 7
    each_side_each_selection_half_minimum: int = 3
    maximum_month_share: float = 0.20
    exact_hold_boundary_gap_share_maximum: float = 0.50
    median_entry_gap_hours_minimum: float = 84.0
    minimum_positive_elapsed_ratio: float = 0.9995
    maximum_invalid_elapsed_run: int = 12
    feature_spearman_absolute_maximum: float = 0.90
    shadow_exposure_absolute_correlation_maximum: float = 0.80
    existing_exposure_absolute_correlation_maximum: float = 0.35
    random_seed: int = 20260720

    @property
    def hold_seconds(self) -> int:
        return self.hold_bars * BAR_SECONDS


def sha256_file(path: str | Path) -> str:
    return prereg.sha256_file(path)


def canonical_hash(payload: Any) -> str:
    return prereg.canonical_hash(payload)


def _maximum_true_run(mask: np.ndarray) -> int:
    maximum = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


class _FenwickMaximum:
    def __init__(self, size: int) -> None:
        self._tree = np.full(size + 1, -1, dtype=np.int64)

    def update(self, index: int, value: int) -> None:
        cursor = index + 1
        while cursor < len(self._tree):
            self._tree[cursor] = max(self._tree[cursor], value)
            cursor += cursor & -cursor

    def prefix_max(self, stop: int) -> int:
        result = -1
        cursor = stop
        while cursor > 0:
            result = max(result, int(self._tree[cursor]))
            cursor -= cursor & -cursor
        return result


def strict_prior_midrank(values: np.ndarray, reference: int) -> np.ndarray:
    """Return exact strict-prior rolling midranks over prior finite values."""
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if reference <= 0:
        raise ValueError("EMFC midrank reference must be positive")
    chronological: deque[float] = deque()
    ordered: list[float] = []
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        current = float(value)
        if len(chronological) == reference:
            less = bisect_left(ordered, current)
            through_equal = bisect_right(ordered, current)
            equal = through_equal - less
            result[index] = (less + 0.5 * equal) / reference
        insort_right(ordered, current)
        chronological.append(current)
        if len(chronological) > reference:
            expired = chronological.popleft()
            ordered.pop(bisect_left(ordered, expired))
    return result


def _joint_state(
    fee_rank: np.ndarray,
    cadence_rank: np.ndarray,
    policy: Policy,
) -> tuple[np.ndarray, np.ndarray]:
    fee_rank = np.asarray(fee_rank, dtype=np.float64)
    cadence_rank = np.asarray(cadence_rank, dtype=np.float64)
    valid = np.isfinite(fee_rank) & np.isfinite(cadence_rank)
    state = np.zeros(len(fee_rank), dtype=np.int8)
    state[
        valid
        & (fee_rank >= policy.upper_rank)
        & (cadence_rank >= policy.upper_rank)
    ] = -1
    state[
        valid
        & (fee_rank <= policy.lower_rank)
        & (cadence_rank <= policy.lower_rank)
    ] = 1
    return state, valid


def _single_state(rank: np.ndarray, policy: Policy) -> tuple[np.ndarray, np.ndarray]:
    rank = np.asarray(rank, dtype=np.float64)
    valid = np.isfinite(rank)
    state = np.zeros(len(rank), dtype=np.int8)
    state[valid & (rank >= policy.upper_rank)] = -1
    state[valid & (rank <= policy.lower_rank)] = 1
    return state, valid


def state_onsets(state: np.ndarray, valid: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.int8)
    valid = np.asarray(valid, dtype=bool)
    if state.shape != valid.shape:
        raise ValueError("EMFC state and validity arrays must have equal shape")
    result = np.zeros(len(state), dtype=bool)
    prior_valid_state: int | None = None
    for index, (current, is_valid) in enumerate(zip(state, valid, strict=True)):
        if not is_valid:
            continue
        current_int = int(current)
        if current_int in {-1, 1} and current_int != prior_valid_state:
            result[index] = True
        prior_valid_state = current_int
    return result


def _validate_policy_binding(artifact: dict[str, Any], policy: Policy) -> None:
    frozen = artifact["policy"]
    checks = {
        "policy_id": frozen["policy_id"] == policy.policy_id,
        "maturity_lag": frozen["source_features"]["origin_height"]
        == f"h-{policy.maturity_lag}",
        "confirmation_blocks": frozen["causal_availability"]["confirmation_blocks"]
        == policy.confirmation_blocks,
        "embargo": frozen["causal_availability"]["historical_embargo_seconds"]
        == policy.historical_embargo_seconds,
        "reference": frozen["normalization"]["reference_valid_heights"]
        == policy.reference_valid_heights,
        "hold": frozen["execution"]["hold_bars"] == policy.hold_bars,
        "entry_delay": frozen["causal_availability"]["publication_latency"]
        == "one complete 5m latency bar"
        and policy.entry_delay_bars == 1
        and frozen["execution"]["bar_size"] == "5m"
        and BAR_SECONDS == 300,
        "rank_tails": frozen["eligibility"]["high_pressure_compressed"]
        == "fee_rank>=0.90 and cadence_rank>=0.90; side=-1 short"
        and frozen["eligibility"]["low_pressure_expanded"]
        == "fee_rank<=0.10 and cadence_rank<=0.10; side=+1 long"
        and policy.upper_rank == 0.90
        and policy.lower_rank == 0.10,
        "daily_reference": frozen["control_construction"][
            "daily_aggregate_shadow"
        ]["normalization"]
        == "strict-prior empirical midrank over exactly 180 valid source days"
        and policy.daily_reference_days == 180,
        "random_seed": frozen["control_construction"]["random_clock"]["seed"]
        == policy.random_seed,
        "integrity_ratio": frozen["source_integrity_gates"][
            "minimum_positive_elapsed_ratio"
        ]
        == policy.minimum_positive_elapsed_ratio,
        "integrity_run": frozen["source_integrity_gates"][
            "maximum_invalid_elapsed_run"
        ]
        == policy.maximum_invalid_elapsed_run,
        "train_min": frozen["event_support_gates"][
            "train_2021_2022_total_minimum"
        ]
        == policy.train_total_minimum,
        "train_max": frozen["event_support_gates"][
            "train_2021_2022_total_maximum"
        ]
        == policy.train_total_maximum,
        "selection_min": frozen["event_support_gates"][
            "selection_2023_total_minimum"
        ]
        == policy.selection_total_minimum,
        "selection_max": frozen["event_support_gates"][
            "selection_2023_total_maximum"
        ]
        == policy.selection_total_maximum,
        "train_each_year": frozen["event_support_gates"][
            "train_each_year_minimum"
        ]
        == policy.train_each_year_minimum,
        "selection_each_half": frozen["event_support_gates"][
            "selection_each_half_minimum"
        ]
        == policy.selection_each_half_minimum,
        "selection_each_quarter": frozen["event_support_gates"][
            "selection_each_quarter_minimum"
        ]
        == policy.selection_each_quarter_minimum,
        "train_side_floor": frozen["event_support_gates"][
            "each_side_each_train_year_minimum"
        ]
        == policy.each_side_each_train_year_minimum,
        "selection_side_floor": frozen["event_support_gates"][
            "each_side_each_selection_half_minimum"
        ]
        == policy.each_side_each_selection_half_minimum,
        "side_share": frozen["event_support_gates"][
            "long_short_share_train_and_selection"
        ]
        == "each side between 25% and 75% inclusive in each window"
        and policy.side_share_minimum == 0.25
        and policy.side_share_maximum == 0.75,
        "month_share": frozen["event_support_gates"]["maximum_month_share"]
        == "<=0.20 separately in train and selection"
        and policy.maximum_month_share == 0.20,
        "hold_gap_share": frozen["event_support_gates"][
            "exact_72h_boundary_gap_share_maximum"
        ]
        == policy.exact_hold_boundary_gap_share_maximum,
        "median_gap": frozen["event_support_gates"][
            "median_entry_gap_hours_minimum"
        ]
        == policy.median_entry_gap_hours_minimum,
        "feature_novelty": frozen["source_novelty_gates"][
            "feature_spearman_absolute_maximum"
        ]
        == policy.feature_spearman_absolute_maximum,
        "shadow_novelty": frozen["source_novelty_gates"][
            "shadow_exposure_absolute_correlation_maximum"
        ]
        == policy.shadow_exposure_absolute_correlation_maximum,
        "existing_novelty": frozen["source_novelty_gates"][
            "existing_network_alpha_exposure_absolute_correlation_maximum"
        ]
        == policy.existing_exposure_absolute_correlation_maximum,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"EMFC support policy differs from preregistration: {failed}")


def load_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("EMFC preregistration file SHA drift")
    artifact = prereg.load_preregistration(PREREGISTRATION)
    if artifact.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("EMFC preregistration manifest hash drift")
    if artifact.get("policy_hash") != PREREGISTRATION_POLICY_HASH:
        raise RuntimeError("EMFC preregistration policy hash drift")
    _validate_policy_binding(artifact, Policy())
    return artifact


def load_source(
    source_csv: str | Path,
    source_manifest: str | Path,
    preregistration: dict[str, Any],
) -> pd.DataFrame:
    source_csv = Path(source_csv)
    source_manifest = Path(source_manifest)
    binding = preregistration["source_manifest"]
    if source_manifest.resolve() != Path(binding["path"]).resolve():
        raise RuntimeError("EMFC source manifest path drift")
    if sha256_file(source_manifest) != binding["sha256"]:
        raise RuntimeError("EMFC source manifest SHA drift")
    output = binding["source_output"]
    if source_csv.resolve() != Path(output["path"]).resolve():
        raise RuntimeError("EMFC source path drift")
    if sha256_file(source_csv) != output["sha256"]:
        raise RuntimeError("EMFC source SHA drift")
    columns = [
        "height",
        "id",
        "previousblockhash",
        "timestamp",
        "mediantime",
        "total_fees",
    ]
    frame = pd.read_csv(
        source_csv,
        usecols=columns,
        dtype={"id": "string", "previousblockhash": "string"},
    )
    if list(frame.columns) != columns:
        raise RuntimeError("EMFC source parser loaded an unexpected schema")
    integer_columns = ["height", "timestamp", "mediantime", "total_fees"]
    for column in integer_columns:
        numeric = pd.to_numeric(frame[column], errors="raise")
        if numeric.isna().any() or not np.all(np.equal(numeric, np.floor(numeric))):
            raise RuntimeError(f"EMFC source {column} must contain exact integers")
        frame[column] = numeric.astype(np.int64)
    frame = frame.sort_values("height").reset_index(drop=True)
    expected_heights = np.arange(
        prereg.FROZEN_START_HEIGHT,
        prereg.FROZEN_END_HEIGHT + 1,
        dtype=np.int64,
    )
    if len(frame) != prereg.FROZEN_ROWS or not np.array_equal(
        frame["height"].to_numpy(np.int64), expected_heights
    ):
        raise RuntimeError("EMFC source is not the exact frozen height range")
    identifiers = frame["id"].astype(str)
    previous = frame["previousblockhash"].astype(str)
    if identifiers.duplicated().any() or not np.array_equal(
        previous.iloc[1:].to_numpy(), identifiers.iloc[:-1].to_numpy()
    ):
        raise RuntimeError("EMFC source hash-chain linkage failed")
    if frame["timestamp"].ge(prereg.FROZEN_END_TIMESTAMP_EXCLUSIVE).any():
        raise RuntimeError("EMFC source crossed the sealed 2024 boundary")
    if frame["total_fees"].lt(0).any():
        raise RuntimeError("EMFC source contains negative total fees")
    return frame


def build_lag_features(
    blocks: pd.DataFrame,
    *,
    lag: int,
    policy: Policy,
) -> pd.DataFrame:
    if lag < 0:
        raise ValueError("EMFC origin lag must be non-negative")
    heights = blocks["height"].to_numpy(np.int64)
    timestamps = blocks["timestamp"].to_numpy(np.int64)
    mediantimes = blocks["mediantime"].to_numpy(np.int64)
    fees = blocks["total_fees"].to_numpy(np.int64)
    candidate_index = np.arange(
        policy.maturity_lag,
        len(blocks) - policy.confirmation_blocks,
        dtype=np.int64,
    )
    origin_index = candidate_index - lag
    valid_origin = origin_index >= 0
    matured_fee = np.full(len(candidate_index), np.nan, dtype=np.float64)
    elapsed = np.full(len(candidate_index), np.nan, dtype=np.float64)
    matured_fee[valid_origin] = fees[origin_index[valid_origin]]
    elapsed[valid_origin] = (
        mediantimes[candidate_index[valid_origin]]
        - mediantimes[origin_index[valid_origin]]
    )
    raw_valid = valid_origin & (matured_fee >= 0.0) & (elapsed > 0.0)
    fee_pressure = np.full(len(candidate_index), np.nan, dtype=np.float64)
    cadence_compression = np.full(len(candidate_index), np.nan, dtype=np.float64)
    fee_pressure[raw_valid] = np.log1p(matured_fee[raw_valid])
    cadence_compression[raw_valid] = -np.log(elapsed[raw_valid] / 60_000.0)
    windows = np.lib.stride_tricks.sliding_window_view(
        timestamps,
        policy.confirmation_blocks + 1,
    )
    raw_available = (
        windows[candidate_index].max(axis=1)
        + policy.historical_embargo_seconds
    )
    decision_boundary = (
        (raw_available + BAR_SECONDS - 1) // BAR_SECONDS
    ) * BAR_SECONDS
    entry_epoch = decision_boundary + policy.entry_delay_bars * BAR_SECONDS
    return pd.DataFrame(
        {
            "maturity_height": heights[candidate_index],
            "confirmation_height": heights[
                candidate_index + policy.confirmation_blocks
            ],
            "maturity_timestamp": timestamps[candidate_index],
            "entry_epoch": entry_epoch,
            "origin_lag": lag,
            "matured_fee_component": matured_fee,
            "maturity_elapsed_seconds": elapsed,
            "raw_valid": raw_valid,
            "fee_pressure": fee_pressure,
            "cadence_compression": cadence_compression,
        }
    )


def add_ranks(features: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    ranked = features.copy()
    ranked["fee_rank"] = strict_prior_midrank(
        ranked["fee_pressure"].to_numpy(np.float64),
        policy.reference_valid_heights,
    )
    ranked["cadence_rank"] = strict_prior_midrank(
        ranked["cadence_compression"].to_numpy(np.float64),
        policy.reference_valid_heights,
    )
    state, valid = _joint_state(
        ranked["fee_rank"].to_numpy(np.float64),
        ranked["cadence_rank"].to_numpy(np.float64),
        policy,
    )
    ranked["state"] = state
    ranked["state_valid"] = valid
    return ranked


def _clock_from_features(
    name: str,
    features: pd.DataFrame,
    state: np.ndarray,
    state_valid: np.ndarray,
    policy: Policy,
    *,
    origin_lag: int | None = None,
) -> pd.DataFrame:
    onsets = state_onsets(state, state_valid)
    candidate_indexes = np.flatnonzero(onsets)
    accepted: list[int] = []
    next_entry = -1
    entries = features["entry_epoch"].to_numpy(np.int64)
    for index in candidate_indexes:
        entry = int(entries[index])
        if entry < next_entry:
            continue
        accepted.append(int(index))
        next_entry = entry + policy.hold_seconds
    if not accepted:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    selected = features.iloc[accepted].copy()
    selected_state = np.asarray(state, dtype=np.int8)[accepted]
    selected.insert(0, "policy_id", policy.policy_id)
    selected.insert(1, "clock", name)
    selected.insert(2, "side", selected_state)
    selected.insert(
        3,
        "state",
        [STATE_LABELS[int(value)] for value in selected_state],
    )
    if origin_lag is not None:
        selected["origin_lag"] = origin_lag
    selected["entry_time"] = pd.to_datetime(
        selected["entry_epoch"], unit="s", utc=True
    )
    selected["exit_time"] = selected["entry_time"] + pd.Timedelta(
        seconds=policy.hold_seconds
    )
    selected = selected.loc[
        selected["exit_time"].gt(GRID_START)
        & selected["entry_time"].lt(GRID_END)
    ]
    return selected[list(CLOCK_COLUMNS)].reset_index(drop=True)


def _primary_and_block_control_clocks(
    primary: pd.DataFrame,
    lag_99: pd.DataFrame,
    lag_101: pd.DataFrame,
    same_height: pd.DataFrame,
    policy: Policy,
) -> dict[str, pd.DataFrame]:
    primary_state = primary["state"].to_numpy(np.int8)
    primary_valid = primary["state_valid"].to_numpy(bool)
    clocks = {
        "primary": _clock_from_features(
            "primary", primary, primary_state, primary_valid, policy
        )
    }
    fee_state, fee_valid = _single_state(
        primary["fee_rank"].to_numpy(np.float64), policy
    )
    cadence_state, cadence_valid = _single_state(
        primary["cadence_rank"].to_numpy(np.float64), policy
    )
    clocks["fee_only"] = _clock_from_features(
        "fee_only", primary, fee_state, fee_valid, policy
    )
    clocks["cadence_only"] = _clock_from_features(
        "cadence_only", primary, cadence_state, cadence_valid, policy
    )
    for name, features, lag in (
        ("same_height_fee", same_height, 0),
        ("pseudo_maturity_99", lag_99, 99),
        ("pseudo_maturity_101", lag_101, 101),
    ):
        clocks[name] = _clock_from_features(
            name,
            features,
            features["state"].to_numpy(np.int8),
            features["state_valid"].to_numpy(bool),
            policy,
            origin_lag=lag,
        )
    return clocks


def _stale_features(primary: pd.DataFrame) -> pd.DataFrame:
    timestamps = primary["maturity_timestamp"].to_numpy(np.int64)
    coordinates = np.unique(timestamps)
    tree = _FenwickMaximum(len(coordinates))
    mapped = np.full(len(primary), -1, dtype=np.int64)
    valid_rank = (
        np.isfinite(primary["fee_rank"].to_numpy(np.float64))
        & np.isfinite(primary["cadence_rank"].to_numpy(np.float64))
    )
    for index, timestamp in enumerate(timestamps):
        target = int(timestamp) - 604_800
        stop = int(np.searchsorted(coordinates, target, side="right"))
        mapped[index] = tree.prefix_max(stop)
        if valid_rank[index]:
            coordinate = int(np.searchsorted(coordinates, timestamp, side="left"))
            tree.update(coordinate, index)
    stale = primary.copy()
    copied_columns = [
        "fee_pressure",
        "cadence_compression",
        "fee_rank",
        "cadence_rank",
    ]
    for column in copied_columns:
        values = np.full(len(primary), np.nan, dtype=np.float64)
        valid_map = mapped >= 0
        values[valid_map] = primary[column].to_numpy(np.float64)[mapped[valid_map]]
        stale[column] = values
    return stale


def _daily_shadow(primary: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    valid = primary.loc[primary["raw_valid"]].copy()
    valid["source_day"] = pd.to_datetime(
        valid["maturity_timestamp"], unit="s", utc=True
    ).dt.floor("D")
    daily = (
        valid.groupby("source_day", sort=True)
        .agg(
            matured_fee_component=("matured_fee_component", "sum"),
            cadence_compression=("cadence_compression", "median"),
            maturity_height=("maturity_height", "max"),
            confirmation_height=("confirmation_height", "max"),
        )
        .reset_index()
    )
    daily["fee_pressure"] = np.log1p(daily["matured_fee_component"])
    daily["fee_rank"] = strict_prior_midrank(
        daily["fee_pressure"].to_numpy(np.float64),
        policy.daily_reference_days,
    )
    daily["cadence_rank"] = strict_prior_midrank(
        daily["cadence_compression"].to_numpy(np.float64),
        policy.daily_reference_days,
    )
    daily["entry_epoch"] = (
        daily["source_day"].astype("int64") // 1_000_000_000
        + 2 * 86_400
        + BAR_SECONDS
    ).astype(np.int64)
    daily["origin_lag"] = policy.maturity_lag
    state, state_valid = _joint_state(
        daily["fee_rank"].to_numpy(np.float64),
        daily["cadence_rank"].to_numpy(np.float64),
        policy,
    )
    return _clock_from_features(
        "daily_aggregate_shadow",
        daily,
        state,
        state_valid,
        policy,
    )


def _derived_primary_clocks(
    primary_clock: pd.DataFrame,
    policy: Policy,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for name, side_transform in (
        ("direction_flip", lambda side: -side),
        ("constant_long_same_clock", lambda side: np.ones(len(side), dtype=int)),
        ("constant_short_same_clock", lambda side: -np.ones(len(side), dtype=int)),
    ):
        frame = primary_clock.copy()
        frame["clock"] = name
        frame["side"] = side_transform(frame["side"].to_numpy(np.int8))
        frame["state"] = frame["side"].map(STATE_LABELS)
        result[name] = frame
    delayed = primary_clock.copy()
    delayed["clock"] = "one_bar_delayed_entry"
    delayed["entry_time"] = pd.to_datetime(delayed["entry_time"], utc=True) + pd.Timedelta(
        seconds=BAR_SECONDS
    )
    delayed["exit_time"] = pd.to_datetime(delayed["exit_time"], utc=True) + pd.Timedelta(
        seconds=BAR_SECONDS
    )
    result["one_bar_delayed_entry"] = delayed
    return result


def _cadence_quartile(values: pd.Series | np.ndarray) -> np.ndarray:
    ranks = np.asarray(values, dtype=np.float64)
    result = np.full(len(ranks), -1, dtype=np.int8)
    finite = np.isfinite(ranks)
    result[finite] = np.minimum((ranks[finite] * 4.0).astype(np.int8), 3)
    return result


def matched_random_clock(
    primary_features: pd.DataFrame,
    primary_clock: pd.DataFrame,
    policy: Policy,
    *,
    maximum_attempts: int = 200,
) -> pd.DataFrame:
    """Build a deterministic year/month/side/activity matched null clock."""
    if primary_clock.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    candidates = primary_features.loc[
        primary_features["state_valid"].to_numpy(bool)
    ].copy()
    candidates["entry_time"] = pd.to_datetime(
        candidates["entry_epoch"], unit="s", utc=True
    )
    candidates = candidates.loc[
        candidates["entry_time"].ge(GRID_START)
        & candidates["entry_time"].lt(GRID_END)
        & ~candidates["maturity_height"].isin(primary_clock["maturity_height"])
    ].copy()
    candidates["activity_quartile"] = _cadence_quartile(
        candidates["cadence_rank"]
    )
    candidates["month"] = candidates["entry_time"].dt.strftime("%Y-%m")
    candidates = candidates.loc[candidates["activity_quartile"].ge(0)]

    targets = primary_clock.copy()
    targets["entry_time"] = pd.to_datetime(targets["entry_time"], utc=True)
    targets = targets.loc[
        targets["entry_time"].ge(GRID_START)
        & targets["entry_time"].lt(GRID_END)
    ].copy()
    if targets.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    targets["month"] = targets["entry_time"].dt.strftime("%Y-%m")
    targets["activity_quartile"] = _cadence_quartile(targets["cadence_rank"])
    target_counts = (
        targets.groupby(["month", "activity_quartile", "side"], sort=True)
        .size()
        .to_dict()
    )

    selected: pd.DataFrame | None = None
    for attempt in range(maximum_attempts):
        rng = np.random.default_rng(policy.random_seed + attempt)
        chosen_rows: list[pd.Series] = []
        next_entry = GRID_START
        failed = False
        for month in sorted(targets["month"].unique()):
            month_targets = targets.loc[targets["month"] == month]
            desired_quartiles = month_targets["activity_quartile"].astype(int).tolist()
            rng.shuffle(desired_quartiles)
            month_candidates = candidates.loc[candidates["month"] == month]
            month_chosen: list[pd.Series] = []
            used_heights: set[int] = set()
            local_next = next_entry
            for quartile in desired_quartiles:
                eligible = month_candidates.loc[
                    month_candidates["activity_quartile"].eq(quartile)
                    & month_candidates["entry_time"].ge(local_next)
                    & ~month_candidates["maturity_height"].isin(used_heights)
                ].sort_values("entry_time")
                if eligible.empty:
                    failed = True
                    break
                early = eligible.iloc[: min(64, len(eligible))]
                chosen = early.iloc[int(rng.integers(0, len(early)))].copy()
                month_chosen.append(chosen)
                used_heights.add(int(chosen["maturity_height"]))
                local_next = pd.Timestamp(chosen["entry_time"]) + pd.Timedelta(
                    seconds=policy.hold_seconds
                )
            if failed:
                break
            month_chosen.sort(key=lambda row: pd.Timestamp(row["entry_time"]))
            sides_by_quartile: dict[int, list[int]] = {}
            for quartile in set(desired_quartiles):
                sides = month_targets.loc[
                    month_targets["activity_quartile"].eq(quartile), "side"
                ].astype(int).tolist()
                rng.shuffle(sides)
                sides_by_quartile[quartile] = sides
            for row in month_chosen:
                quartile = int(row["activity_quartile"])
                row["side"] = sides_by_quartile[quartile].pop()
                chosen_rows.append(row)
            if month_chosen:
                next_entry = pd.Timestamp(month_chosen[-1]["entry_time"]) + pd.Timedelta(
                    seconds=policy.hold_seconds
                )
        if not failed and len(chosen_rows) == len(targets):
            selected = pd.DataFrame(chosen_rows).sort_values("entry_time")
            break
    if selected is None:
        raise RuntimeError("EMFC matched random clock could not satisfy frozen strata")

    selected_counts = (
        selected.assign(
            month=pd.to_datetime(selected["entry_time"], utc=True).dt.strftime("%Y-%m"),
            activity_quartile=_cadence_quartile(selected["cadence_rank"]),
        )
        .groupby(["month", "activity_quartile", "side"], sort=True)
        .size()
        .to_dict()
    )
    if selected_counts != target_counts:
        raise RuntimeError("EMFC matched random clock stratum counts changed")
    entries = pd.to_datetime(selected["entry_time"], utc=True)
    if len(entries) > 1 and (
        entries.iloc[1:].reset_index(drop=True)
        < (
            entries.iloc[:-1].reset_index(drop=True)
            + pd.Timedelta(seconds=policy.hold_seconds)
        )
    ).any():
        raise RuntimeError("EMFC matched random clock overlaps")
    selected.insert(0, "policy_id", policy.policy_id)
    selected.insert(1, "clock", "year_month_side_activity_stratified_random_clock")
    selected["state"] = selected["side"].map(STATE_LABELS)
    selected["exit_time"] = pd.to_datetime(
        selected["entry_time"], utc=True
    ) + pd.Timedelta(seconds=policy.hold_seconds)
    return selected[list(CLOCK_COLUMNS)].reset_index(drop=True)


def _spearman(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return 0.0
    value = pd.Series(left[mask]).corr(pd.Series(right[mask]), method="spearman")
    return 0.0 if value is None or not np.isfinite(value) else float(value)


def feature_novelty_summary(
    primary: pd.DataFrame,
    same_height: pd.DataFrame,
    lag_99: pd.DataFrame,
    lag_101: pd.DataFrame,
    policy: Policy,
) -> dict[str, Any]:
    fee_mask = primary["raw_valid"].to_numpy(bool) & same_height[
        "raw_valid"
    ].to_numpy(bool)
    primary_state = primary["state"].to_numpy(np.int8)
    values: dict[str, float] = {
        "fee_pressure_vs_same_height_fee": _spearman(
            primary["fee_pressure"].to_numpy(np.float64),
            same_height["fee_pressure"].to_numpy(np.float64),
            fee_mask,
        )
    }
    for name, comparison in (
        ("primary_state_vs_pseudo_maturity_99", lag_99),
        ("primary_state_vs_pseudo_maturity_101", lag_101),
    ):
        comparison_state = comparison["state"].to_numpy(np.int8)
        jointly_valid = primary["state_valid"].to_numpy(bool) & comparison[
            "state_valid"
        ].to_numpy(bool)
        extreme_union = jointly_valid & (
            (primary_state != 0) | (comparison_state != 0)
        )
        values[name] = _spearman(primary_state, comparison_state, extreme_union)
    checks = {
        name: abs(value) <= policy.feature_spearman_absolute_maximum
        for name, value in values.items()
    }
    return {
        "method": "Spearman; pseudo-state comparisons use the jointly-valid union where either state is extreme",
        "absolute_maximum": policy.feature_spearman_absolute_maximum,
        "correlations": values,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _clock_exposure(clock: pd.DataFrame) -> np.ndarray:
    grid_rows = int((GRID_END - GRID_START).total_seconds() // BAR_SECONDS)
    exposure = np.zeros(grid_rows, dtype=np.float64)
    if clock.empty:
        return exposure
    entries = pd.to_datetime(clock["entry_time"], utc=True)
    exits = pd.to_datetime(clock["exit_time"], utc=True)
    sides = pd.to_numeric(clock["side"], errors="raise").to_numpy(np.float64)
    for entry, exit_time, side in zip(entries, exits, sides, strict=True):
        start = max(0, int((entry - GRID_START).total_seconds() // BAR_SECONDS))
        stop = min(
            grid_rows,
            int(math.ceil((exit_time - GRID_START).total_seconds() / BAR_SECONDS)),
        )
        if stop > start:
            exposure[start:stop] += side
    return exposure


def _exposure_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right):
        raise ValueError("EMFC exposure vectors must have equal length")
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return 0.0 if not np.isfinite(value) else value


def _load_comparator_clock(definition: dict[str, str]) -> pd.DataFrame:
    path = Path(definition["path"])
    if sha256_file(path) != definition["sha256"]:
        raise RuntimeError(f"EMFC novelty comparator drifted: {path}")
    usecols = [
        definition["entry_column"],
        definition["exit_column"],
        definition["side_column"],
    ]
    header = pd.read_csv(path, nrows=0)
    if not set(usecols).issubset(header.columns):
        raise RuntimeError(f"EMFC novelty comparator schema drifted: {path}")
    frame = pd.read_csv(path, usecols=usecols)
    return frame.rename(
        columns={
            definition["entry_column"]: "entry_time",
            definition["exit_column"]: "exit_time",
            definition["side_column"]: "side",
        }
    )


def exposure_novelty_summary(
    clocks: dict[str, pd.DataFrame],
    preregistration: dict[str, Any],
    policy: Policy,
) -> dict[str, Any]:
    primary_exposure = _clock_exposure(clocks["primary"])
    shadow_names = preregistration["policy"]["source_novelty_gates"][
        "shadow_controls"
    ]
    missing_shadows = sorted(set(shadow_names) - set(clocks))
    if missing_shadows:
        raise RuntimeError(
            f"EMFC frozen novelty shadows are missing: {missing_shadows}"
        )
    shadow_correlations = {
        name: _exposure_correlation(primary_exposure, _clock_exposure(clocks[name]))
        for name in shadow_names
    }
    comparator_correlations: dict[str, float] = {}
    for name, definition in preregistration["novelty_comparators"].items():
        comparator = _load_comparator_clock(definition)
        comparator_correlations[name] = _exposure_correlation(
            primary_exposure,
            _clock_exposure(comparator),
        )
    shadow_checks = {
        name: abs(value) <= policy.shadow_exposure_absolute_correlation_maximum
        for name, value in shadow_correlations.items()
    }
    comparator_checks = {
        name: abs(value) <= policy.existing_exposure_absolute_correlation_maximum
        for name, value in comparator_correlations.items()
    }
    return {
        "grid_start": GRID_START.isoformat(),
        "grid_end_exclusive": GRID_END.isoformat(),
        "bar_seconds": BAR_SECONDS,
        "shadow_absolute_maximum": policy.shadow_exposure_absolute_correlation_maximum,
        "existing_alpha_absolute_maximum": policy.existing_exposure_absolute_correlation_maximum,
        "shadow_correlations": shadow_correlations,
        "existing_alpha_correlations": comparator_correlations,
        "checks": {**shadow_checks, **{f"existing_{k}": v for k, v in comparator_checks.items()}},
        "passed": bool(all(shadow_checks.values()) and all(comparator_checks.values())),
    }


def _window(clock: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    entry = pd.to_datetime(clock["entry_time"], utc=True)
    return clock.loc[
        entry.ge(pd.Timestamp(start, tz="UTC"))
        & entry.lt(pd.Timestamp(end, tz="UTC"))
    ].copy()


def _month_share(clock: pd.DataFrame) -> float:
    if clock.empty:
        return 1.0
    months = pd.to_datetime(clock["entry_time"], utc=True).dt.strftime("%Y-%m")
    return float(months.value_counts().max() / len(clock))


def _side_share_ok(clock: pd.DataFrame, policy: Policy) -> bool:
    if clock.empty:
        return False
    share_long = float((clock["side"] == 1).mean())
    return policy.side_share_minimum <= share_long <= policy.side_share_maximum


def event_support_summary(clock: pd.DataFrame, policy: Policy) -> dict[str, Any]:
    windows = {
        "train": _window(clock, "2021-01-01", "2023-01-01"),
        "train_2021": _window(clock, "2021-01-01", "2022-01-01"),
        "train_2022": _window(clock, "2022-01-01", "2023-01-01"),
        "selection": _window(clock, "2023-01-01", "2024-01-01"),
        "selection_h1": _window(clock, "2023-01-01", "2023-07-01"),
        "selection_h2": _window(clock, "2023-07-01", "2024-01-01"),
        "selection_q1": _window(clock, "2023-01-01", "2023-04-01"),
        "selection_q2": _window(clock, "2023-04-01", "2023-07-01"),
        "selection_q3": _window(clock, "2023-07-01", "2023-10-01"),
        "selection_q4": _window(clock, "2023-10-01", "2024-01-01"),
    }
    counts = {name: int(len(frame)) for name, frame in windows.items()}
    side_counts = {
        name: {
            "long": int((frame["side"] == 1).sum()),
            "short": int((frame["side"] == -1).sum()),
        }
        for name, frame in windows.items()
    }
    entries = pd.to_datetime(clock["entry_time"], utc=True).sort_values()
    gaps = entries.diff().dropna().dt.total_seconds().to_numpy(np.float64)
    exact_gap_share = float(np.mean(gaps == policy.hold_seconds)) if len(gaps) else 1.0
    median_gap_hours = float(np.median(gaps) / 3_600.0) if len(gaps) else 0.0
    checks = {
        "train_total_minimum": counts["train"] >= policy.train_total_minimum,
        "train_total_maximum": counts["train"] <= policy.train_total_maximum,
        "train_each_year": all(
            counts[name] >= policy.train_each_year_minimum
            for name in ("train_2021", "train_2022")
        ),
        "train_each_side_each_year": all(
            side_counts[year][side] >= policy.each_side_each_train_year_minimum
            for year in ("train_2021", "train_2022")
            for side in ("long", "short")
        ),
        "train_side_share": _side_share_ok(windows["train"], policy),
        "selection_total_minimum": counts["selection"]
        >= policy.selection_total_minimum,
        "selection_total_maximum": counts["selection"]
        <= policy.selection_total_maximum,
        "selection_each_half": all(
            counts[name] >= policy.selection_each_half_minimum
            for name in ("selection_h1", "selection_h2")
        ),
        "selection_each_quarter": all(
            counts[name] >= policy.selection_each_quarter_minimum
            for name in (
                "selection_q1",
                "selection_q2",
                "selection_q3",
                "selection_q4",
            )
        ),
        "selection_each_side_each_half": all(
            side_counts[half][side]
            >= policy.each_side_each_selection_half_minimum
            for half in ("selection_h1", "selection_h2")
            for side in ("long", "short")
        ),
        "selection_side_share": _side_share_ok(windows["selection"], policy),
        "train_month_share": _month_share(windows["train"])
        <= policy.maximum_month_share,
        "selection_month_share": _month_share(windows["selection"])
        <= policy.maximum_month_share,
        "exact_hold_boundary_gap_share": exact_gap_share
        <= policy.exact_hold_boundary_gap_share_maximum,
        "median_entry_gap": median_gap_hours
        >= policy.median_entry_gap_hours_minimum,
    }
    return {
        "counts": counts,
        "side_counts": side_counts,
        "train_maximum_month_share": _month_share(windows["train"]),
        "selection_maximum_month_share": _month_share(windows["selection"]),
        "exact_hold_boundary_gap_share": exact_gap_share,
        "median_entry_gap_hours": median_gap_hours,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def source_integrity_summary(
    blocks: pd.DataFrame,
    primary: pd.DataFrame,
    policy: Policy,
) -> dict[str, Any]:
    raw_valid = primary["raw_valid"].to_numpy(bool)
    positive_ratio = float(raw_valid.mean()) if len(raw_valid) else 0.0
    invalid_run = _maximum_true_run(~raw_valid)
    checks = {
        "exact_candidate_heights": len(primary) == 212_989,
        "positive_elapsed_ratio": positive_ratio
        >= policy.minimum_positive_elapsed_ratio,
        "maximum_invalid_elapsed_run": invalid_run
        <= policy.maximum_invalid_elapsed_run,
        "confirmation_containment": bool(
            len(primary)
            and primary["confirmation_height"].max()
            <= prereg.FROZEN_END_HEIGHT
        ),
        "nonnegative_total_fees": bool(blocks["total_fees"].ge(0).all()),
        "pre_2024": bool(
            blocks["timestamp"].lt(prereg.FROZEN_END_TIMESTAMP_EXCLUSIVE).all()
        ),
    }
    return {
        "source_rows": int(len(blocks)),
        "candidate_heights": int(len(primary)),
        "valid_candidate_heights": int(raw_valid.sum()),
        "positive_elapsed_ratio": positive_ratio,
        "maximum_invalid_elapsed_run": invalid_run,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)


def _atomic_write_clock(path: str | Path, clock: pd.DataFrame) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        clock.to_csv(
            temporary,
            index=False,
            columns=list(CLOCK_COLUMNS),
            date_format="%Y-%m-%dT%H:%M:%SZ",
            float_format="%.17g",
            lineterminator="\n",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)


def _validate_output_paths(
    source_csv: str | Path,
    source_manifest: str | Path,
    output: str | Path,
    clock_output: str | Path,
    preregistration: dict[str, Any],
) -> None:
    protected = {
        Path(source_csv).resolve(),
        Path(source_manifest).resolve(),
        PREREGISTRATION.resolve(),
        prereg.MECHANISM_DECISION.resolve(),
        prereg.PREREGISTRATION_SOURCE.resolve(),
        SUPPORT_SOURCE.resolve(),
        *(
            Path(definition["path"]).resolve()
            for definition in preregistration["novelty_comparators"].values()
        ),
    }
    outputs = {Path(output).resolve(), Path(clock_output).resolve()}
    if len(outputs) != 2 or outputs & protected:
        raise ValueError("EMFC support outputs must be distinct from protected inputs")


def run(
    *,
    source_csv: str | Path = DEFAULT_SOURCE,
    source_manifest: str | Path = DEFAULT_SOURCE_MANIFEST,
    output: str | Path = DEFAULT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> dict[str, Any]:
    preregistration = load_preregistration()
    _validate_output_paths(
        source_csv,
        source_manifest,
        output,
        clock_output,
        preregistration,
    )
    policy = Policy()
    blocks = load_source(source_csv, source_manifest, preregistration)
    outcome_boundary = {
        **OUTCOME_BOUNDARY,
        "post_2023_source_rows_loaded": int(
            blocks["timestamp"].ge(prereg.FROZEN_END_TIMESTAMP_EXCLUSIVE).sum()
        ),
    }
    outcome_boundary_all_zero = bool(
        all(value == 0 for value in outcome_boundary.values())
    )
    primary = add_ranks(
        build_lag_features(blocks, lag=policy.maturity_lag, policy=policy),
        policy,
    )
    lag_99 = add_ranks(build_lag_features(blocks, lag=99, policy=policy), policy)
    lag_101 = add_ranks(build_lag_features(blocks, lag=101, policy=policy), policy)
    same_height = build_lag_features(blocks, lag=0, policy=policy)
    maturity_positions = (
        same_height["maturity_height"].to_numpy(np.int64)
        - prereg.FROZEN_START_HEIGHT
    )
    same_height_fees = blocks["total_fees"].to_numpy(np.int64)[maturity_positions]
    same_height["matured_fee_component"] = same_height_fees
    same_height["fee_pressure"] = np.log1p(same_height_fees.astype(np.float64))
    same_height["maturity_elapsed_seconds"] = primary[
        "maturity_elapsed_seconds"
    ]
    same_height["cadence_compression"] = primary["cadence_compression"]
    same_height["raw_valid"] = primary["raw_valid"].to_numpy(bool)
    same_height = add_ranks(same_height, policy)

    integrity = source_integrity_summary(blocks, primary, policy)
    feature_novelty = feature_novelty_summary(
        primary,
        same_height,
        lag_99,
        lag_101,
        policy,
    )
    clocks = _primary_and_block_control_clocks(
        primary,
        lag_99,
        lag_101,
        same_height,
        policy,
    )
    stale = _stale_features(primary)
    stale_state, stale_valid = _joint_state(
        stale["fee_rank"].to_numpy(np.float64),
        stale["cadence_rank"].to_numpy(np.float64),
        policy,
    )
    clocks["stale_7d"] = _clock_from_features(
        "stale_7d", stale, stale_state, stale_valid, policy
    )
    clocks["daily_aggregate_shadow"] = _daily_shadow(primary, policy)
    event_support = event_support_summary(clocks["primary"], policy)
    exposure_novelty = exposure_novelty_summary(clocks, preregistration, policy)
    clocks[
        "year_month_side_activity_stratified_random_clock"
    ] = matched_random_clock(primary, clocks["primary"], policy)
    clocks.update(_derived_primary_clocks(clocks["primary"], policy))
    combined_clock = pd.concat(clocks.values(), ignore_index=True).sort_values(
        ["clock", "entry_time"], kind="stable"
    )
    _atomic_write_clock(clock_output, combined_clock)
    all_checks = {
        "outcome_boundary_all_zero": outcome_boundary_all_zero,
        **{f"integrity_{key}": value for key, value in integrity["checks"].items()},
        **{f"support_{key}": value for key, value in event_support["checks"].items()},
        **{
            f"feature_novelty_{key}": value
            for key, value in feature_novelty["checks"].items()
        },
        **{
            f"exposure_novelty_{key}": value
            for key, value in exposure_novelty["checks"].items()
        },
    }
    passed = bool(all(all_checks.values()))
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy": asdict(policy),
        "outcomes_opened": False,
        "outcome_boundary": outcome_boundary,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
            "policy_hash": PREREGISTRATION_POLICY_HASH,
        },
        "source": {
            "path": str(source_csv),
            "sha256": sha256_file(source_csv),
            "manifest_path": str(source_manifest),
            "manifest_sha256": sha256_file(source_manifest),
            "rows_loaded": int(len(blocks)),
            "columns_loaded": [
                "height",
                "id",
                "previousblockhash",
                "timestamp",
                "mediantime",
                "total_fees",
            ],
        },
        "source_integrity": integrity,
        "feature_novelty": feature_novelty,
        "event_support": event_support,
        "exposure_novelty": exposure_novelty,
        "control_roles": {
            "source_novelty_shadows": preregistration["policy"][
                "source_novelty_gates"
            ]["shadow_controls"],
            "performance_nulls_not_used_as_novelty_failures": [
                "direction_flip",
                "constant_long_same_clock",
                "constant_short_same_clock",
                "year_month_side_activity_stratified_random_clock",
                "one_bar_delayed_entry",
            ],
        },
        "clock": {
            "path": str(clock_output),
            "sha256": sha256_file(clock_output),
            "rows": int(len(combined_clock)),
            "rows_by_clock": {
                name: int(len(frame)) for name, frame in sorted(clocks.items())
            },
            "primary_first_entry": (
                str(clocks["primary"]["entry_time"].min())
                if len(clocks["primary"])
                else None
            ),
            "primary_last_entry": (
                str(clocks["primary"]["entry_time"].max())
                if len(clocks["primary"])
                else None
            ),
        },
        "support_gate": {
            "checks": all_checks,
            "passed": passed,
        },
        "support_source": {
            "path": str(SUPPORT_SOURCE),
            "sha256": sha256_file(SUPPORT_SOURCE),
        },
        "sealed_outcomes": ["2021", "2022", "2023", "2024+"],
        "failure_action": (
            "freeze a strict train evaluator before opening any market or funding value"
            if passed
            else "reject EMFC-864 without opening any market or funding outcome and do not repair the singleton"
        ),
    }
    result = {**core, "result_hash": canonical_hash(core)}
    _atomic_write_json(output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", default=str(DEFAULT_SOURCE))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK_OUTPUT))
    return parser.parse_args()


def main() -> None:
    result = run(**vars(parse_args()))
    print(
        json.dumps(
            {
                "outcomes_opened": result["outcomes_opened"],
                "source_integrity": result["source_integrity"],
                "feature_novelty": result["feature_novelty"],
                "event_support": result["event_support"],
                "exposure_novelty": result["exposure_novelty"],
                "clock": result["clock"],
                "support_gate": result["support_gate"],
                "result_hash": result["result_hash"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
