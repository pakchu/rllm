"""Fit and roll out the frozen 31-member BCTP cheap-policy family.

This module consumes source-only states and already-created transition rewards.
It has no market or funding loader.  The legacy BCRT clock is source-only and
is used solely to construct the two frozen six-hour comparator schedules.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
import pandas as pd

from training import bctp_cheap_policies as cheap
from training import freeze_block_clearing_target_position_evaluator as freeze
from training import preregister_block_clearing_relational_topology as bcrt
from training import preregister_block_clearing_target_position_mdp as prereg


PROMOTABLE_PRIMARY_IDS: Final = (
    "categorical_linear_fqi",
    "categorical_ridge_fqi",
    "extra_trees_fqi",
)
ALGORITHM_BY_PRIMARY: Final = {
    "categorical_linear_fqi": "linear",
    "categorical_ridge_fqi": "ridge",
    "extra_trees_fqi": "extra_trees",
}
SCHEDULE_COLUMNS: Final = (
    "policy_id",
    "sequence_id",
    "entry_time",
    "target",
)
BCRT_CLOCK_COLUMNS: Final = (
    "signal_id",
    "bucket_start",
    "signal_available_time",
    "entry_time",
    "exit_time",
    *bcrt.TOKEN_COLUMNS,
)
FIVE_MINUTES: Final = pd.Timedelta(minutes=5)
SIX_HOURS: Final = pd.Timedelta(hours=6)


@dataclass(frozen=True)
class BcrtClockComparator:
    target: str


@dataclass(frozen=True)
class FittedFamily:
    policies: OrderedDict[str, Any]
    fitted_estimators: int
    memory_tables_fit: int
    state_rows: int


def _utc(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"BCTP {name} must be timezone aware")
    if pd.isna(timestamp):
        raise ValueError(f"BCTP {name} is NaT")
    return timestamp.tz_convert("UTC")


def _iso_z(value: Any) -> str:
    return _utc(value, name="output timestamp").isoformat().replace(
        "+00:00",
        "Z",
    )


def _stage_bounds(stage: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if stage not in {"2020", "2021", "2022", "2023"}:
        raise ValueError(f"BCTP unsupported stage: {stage!r}")
    start = pd.Timestamp(f"{stage}-01-01T00:00:00Z")
    end = pd.Timestamp(f"{int(stage) + 1}-01-01T00:00:00Z")
    return start, end


def _validate_training_arrays(
    states: Any,
    reward_tensor: Any,
    terminal: Any,
    reachable_mask: Any,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    records = cheap.validate_source_state_schema(
        states,
        require_known=True,
    )
    rewards = np.asarray(reward_tensor, dtype=np.float64)
    terminal_array = np.asarray(terminal, dtype=bool)
    reachable = np.asarray(reachable_mask, dtype=bool)
    count = len(records)
    if rewards.shape != (count, 3, 3):
        raise ValueError("BCTP family reward tensor shape changed")
    if terminal_array.shape != (count,) or reachable.shape != (count, 3):
        raise ValueError("BCTP family transition masks changed")
    if not np.isfinite(rewards[reachable]).all():
        raise ValueError("BCTP family reachable rewards are non-finite")
    if not terminal_array[-1] or terminal_array[:-1].any():
        raise ValueError("BCTP family terminal mask changed")
    expected_reachable = np.ones_like(reachable)
    expected_reachable[0] = False
    expected_reachable[0, cheap.POSITIONS.index("POSITION_FLAT")] = True
    if not np.array_equal(reachable, expected_reachable):
        raise ValueError("BCTP family reachability changed")
    return records, rewards, terminal_array, reachable


def fit_family(
    states: Any,
    reward_tensor: Any,
    terminal: Any,
    reachable_mask: Any,
) -> FittedFamily:
    """Fit every frozen learned/control policy without outcome-file access."""

    records, rewards, terminal_array, reachable = _validate_training_arrays(
        states,
        reward_tensor,
        terminal,
        reachable_mask,
    )
    policies: dict[str, Any] = {
        "always_flat": cheap.constant_policy("TARGET_FLAT"),
        "always_long": cheap.constant_policy("TARGET_LONG"),
        "always_short": cheap.constant_policy("TARGET_SHORT"),
        "previous_target_persistence": cheap.persistence_policy(),
        "exact_signature_memory": cheap.exact_signature_memory_policy(
            records,
            rewards,
            reachable,
        ),
        "bcrt_exact_six_hour_always_long": BcrtClockComparator(
            "TARGET_LONG"
        ),
        "bcrt_exact_six_hour_always_short": BcrtClockComparator(
            "TARGET_SHORT"
        ),
    }
    fitted_estimators = 0
    for primary_id, algorithm in ALGORITHM_BY_PRIMARY.items():
        primary = cheap.fit_fitted_q(
            records,
            rewards,
            terminal_array,
            reachable,
            algorithm=algorithm,
        )
        policies[primary_id] = primary
        fitted_estimators += 1
        for suffix, feature_mode in (
            ("current_only", "current_only"),
            ("reversed_sequence", "reversed"),
            ("masked_source", "masked"),
        ):
            policies[f"{primary_id}_{suffix}"] = cheap.fit_fitted_q(
                records,
                rewards,
                terminal_array,
                reachable,
                algorithm=algorithm,
                feature_mode=feature_mode,
            )
            fitted_estimators += 1
        for suffix, reward_mode in (
            ("shuffled_reward", "shuffled_month"),
            ("circular_21_reward", "circular_21"),
        ):
            policies[f"{primary_id}_{suffix}"] = cheap.fit_fitted_q(
                records,
                rewards,
                terminal_array,
                reachable,
                algorithm=algorithm,
                reward_mode=reward_mode,
            )
            fitted_estimators += 1
        policies[f"{primary_id}_direction_flip"] = cheap.direction_flip(
            primary
        )
        policies[
            f"{primary_id}_action_code_permutation"
        ] = cheap.action_code_permutation_policy(
            records,
            rewards,
            terminal_array,
            reachable,
            algorithm=algorithm,
        )
        fitted_estimators += 1
    ordered = OrderedDict(
        (policy_id, policies[policy_id])
        for policy_id in freeze.FAMILY_IDS
    )
    if tuple(ordered) != freeze.FAMILY_IDS or len(ordered) != 31:
        raise RuntimeError("BCTP frozen family construction changed")
    return FittedFamily(
        policies=ordered,
        fitted_estimators=fitted_estimators,
        memory_tables_fit=1,
        state_rows=len(records),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_bcrt_clock_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != BCRT_CLOCK_COLUMNS:
        raise ValueError("BCTP BCRT comparator clock schema changed")
    entries = [_utc(value, name="BCRT entry") for value in frame["entry_time"]]
    exits = [_utc(value, name="BCRT exit") for value in frame["exit_time"]]
    if (
        len(set(frame["signal_id"])) != len(frame)
        or any(exit_time - entry != SIX_HOURS for entry, exit_time in zip(entries, exits))
        or any(
            next_entry < exit_time
            for exit_time, next_entry in zip(exits, entries[1:])
        )
    ):
        raise ValueError("BCTP BCRT comparator clock integrity changed")
    output = frame.copy()
    output["entry_time"] = pd.DatetimeIndex(entries)
    output["exit_time"] = pd.DatetimeIndex(exits)
    return output


def load_bcrt_clock(
    path: str | Path = freeze.BCRT_CLOCK,
    *,
    allow_synthetic_clock: bool = False,
) -> pd.DataFrame:
    requested = Path(path)
    if not requested.is_absolute():
        requested = freeze.REPOSITORY_ROOT / requested
    frozen = freeze.REPOSITORY_ROOT / freeze.BCRT_CLOCK
    is_frozen = requested.resolve() == frozen.resolve()
    if not is_frozen and not allow_synthetic_clock:
        raise ValueError("BCTP comparator clock must match the frozen BCRT clock")
    if allow_synthetic_clock and is_frozen:
        raise ValueError("BCTP synthetic-clock override was unnecessary")
    if is_frozen and _file_sha256(requested) != freeze.EXPECTED_STATIC_SHA256[
        str(freeze.BCRT_CLOCK)
    ]:
        raise ValueError("BCTP frozen BCRT clock hash changed")
    frame = pd.read_csv(requested, compression="gzip", dtype=str)
    return _validate_bcrt_clock_frame(frame)


def _source_schedule(
    policy_id: str,
    policy: Any,
    states: pd.DataFrame,
) -> pd.DataFrame:
    if tuple(states.columns) != prereg.SOURCE_SEQUENCE_COLUMNS:
        raise ValueError("BCTP target source-state schema changed")
    targets = cheap.rollout_policy(policy, states)
    rows = [
        {
            "policy_id": policy_id,
            "sequence_id": str(sequence_id),
            "entry_time": _iso_z(entry_time),
            "target": target,
        }
        for sequence_id, entry_time, target in zip(
            states["sequence_id"],
            states["entry_time"],
            targets,
        )
    ]
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def _bcrt_schedule(
    policy_id: str,
    target: str,
    clock: pd.DataFrame,
    *,
    stage: str,
) -> pd.DataFrame:
    start, end = _stage_bounds(stage)
    terminal = end - FIVE_MINUTES
    selected = clock.loc[
        clock["entry_time"].ge(start) & clock["entry_time"].lt(terminal)
    ]
    events: list[tuple[pd.Timestamp, int, dict[str, str]]] = []
    for row in selected.itertuples(index=False):
        entry = _utc(row.entry_time, name="BCRT entry")
        exit_time = _utc(row.exit_time, name="BCRT exit")
        if exit_time > terminal:
            raise ValueError("BCTP BCRT comparator escapes terminal flatten")
        events.append(
            (
                entry,
                1,
                {
                "policy_id": policy_id,
                "sequence_id": f"{row.signal_id}:entry",
                "entry_time": _iso_z(entry),
                "target": target,
                },
            )
        )
        if exit_time < terminal:
            events.append(
                (
                    exit_time,
                    0,
                    {
                    "policy_id": policy_id,
                    "sequence_id": f"{row.signal_id}:exit",
                    "entry_time": _iso_z(exit_time),
                    "target": "TARGET_FLAT",
                    },
                )
            )
    events.sort(key=lambda item: (item[0], item[1]))
    rows: list[dict[str, str]] = []
    for _, _, event in events:
        if rows and rows[-1]["entry_time"] == event["entry_time"]:
            rows[-1] = event
        else:
            rows.append(event)
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def _delay_schedule(frame: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    _, end = _stage_bounds(stage)
    terminal = end - FIVE_MINUTES
    delayed = frame.copy()
    times = pd.DatetimeIndex(
        [_utc(value, name="base schedule time") for value in frame["entry_time"]]
    ) + FIVE_MINUTES
    keep = times < terminal
    delayed = delayed.loc[keep].copy()
    delayed["entry_time"] = [_iso_z(value) for value in times[keep]]
    delayed["sequence_id"] = (
        delayed["sequence_id"].astype(str) + ":delay_5m"
    )
    return delayed.reset_index(drop=True)


def build_transfer_schedules(
    family: FittedFamily,
    target_states: pd.DataFrame,
    *,
    stage: str,
    bcrt_clock: pd.DataFrame | None = None,
) -> tuple[
    OrderedDict[str, pd.DataFrame],
    OrderedDict[str, pd.DataFrame],
]:
    """Roll out base family and +5m primary schedules before stage outcomes."""

    start, end = _stage_bounds(stage)
    states = target_states.copy()
    states["entry_time"] = [
        _utc(value, name="target source entry")
        for value in states["entry_time"]
    ]
    terminal = end - FIVE_MINUTES
    if (
        states.empty
        or not states["entry_time"].is_monotonic_increasing
        or states["entry_time"].duplicated().any()
        or not states["entry_time"].between(
            start,
            terminal,
            inclusive="left",
        ).all()
    ):
        raise ValueError("BCTP target source stage changed")
    clock = (
        _validate_bcrt_clock_frame(bcrt_clock.copy())
        if bcrt_clock is not None
        else load_bcrt_clock()
    )
    base: OrderedDict[str, pd.DataFrame] = OrderedDict()
    for policy_id, policy in family.policies.items():
        if isinstance(policy, BcrtClockComparator):
            schedule = _bcrt_schedule(
                policy_id,
                policy.target,
                clock,
                stage=stage,
            )
        else:
            schedule = _source_schedule(policy_id, policy, states)
        if schedule.empty:
            raise ValueError(f"BCTP schedule is empty: {policy_id}")
        base[policy_id] = schedule
    if tuple(base) != freeze.FAMILY_IDS:
        raise RuntimeError("BCTP transfer family order changed")

    delayed = OrderedDict(
        (
            primary_id,
            _delay_schedule(base[primary_id], stage=stage),
        )
        for primary_id in PROMOTABLE_PRIMARY_IDS
    )
    for primary_id in PROMOTABLE_PRIMARY_IDS:
        permutation_id = f"{primary_id}_action_code_permutation"
        canonical = base[primary_id].loc[
            :,
            ["sequence_id", "entry_time", "target"],
        ].reset_index(drop=True)
        permuted = base[permutation_id].loc[
            :,
            ["sequence_id", "entry_time", "target"],
        ].reset_index(drop=True)
        if not canonical.equals(permuted):
            raise RuntimeError(
                "BCTP neutral action-code permutation changed schedule"
            )
    return base, delayed


__all__ = [
    "PROMOTABLE_PRIMARY_IDS",
    "SCHEDULE_COLUMNS",
    "BCRT_CLOCK_COLUMNS",
    "BcrtClockComparator",
    "FittedFamily",
    "fit_family",
    "load_bcrt_clock",
    "build_transfer_schedules",
]
