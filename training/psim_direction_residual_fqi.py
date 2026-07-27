"""Direction-residual reward and policy core for PSIM RLLM2 S5."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from training import (
    preregister_psim_d8_rllm2_s5_direction_residual_ridge_fqi as prereg,
)
from training import psim_semantic_fqi_policies as fqi


SCHEDULE_COLUMNS = ("policy_id", "sequence_id", "entry_time", "target")
RESIDUAL_LEDGER_COLUMNS = (
    "sequence_id",
    "entry_time",
    "current_position",
    "action_name",
    "action_target",
    "executed_target",
    "reachable",
    "terminal",
    "reward",
    "direction_residual_delta_applied",
    "direction_residual_reward",
    "multiplier",
    "held_path_downside_fraction",
    "changed_notional_fraction",
    "entry_cost",
    "terminal_cost",
    "funding_cash",
    "bars_held",
)
EXPECTED_LEDGER_COLUMNS = tuple(
    column
    for column in RESIDUAL_LEDGER_COLUMNS
    if column
    not in {
        "direction_residual_delta_applied",
        "direction_residual_reward",
    }
)
FEATURE_NAME_BY_POLICY = {
    prereg.PRIMARY_POLICY_ID: "semantic",
    f"{prereg.PRIMARY_POLICY_ID}_action_code_permutation": "semantic",
    f"{prereg.PRIMARY_POLICY_ID}_direction_flip": "semantic",
    f"{prereg.PRIMARY_POLICY_ID}_circular_21_reward": "semantic",
    f"{prereg.PRIMARY_POLICY_ID}_within_month_shuffled_reward": "semantic",
    f"{prereg.PRIMARY_POLICY_ID}_current_position_only": (
        "current_position_only"
    ),
    f"{prereg.PRIMARY_POLICY_ID}_masked_semantic_embedding": (
        "masked_semantic_embedding"
    ),
    f"{prereg.PRIMARY_POLICY_ID}_metadata_frontmatter_only": (
        "metadata_frontmatter_only"
    ),
}
DEGENERATE_CONTROL_IDS = (
    "always_flat",
    "always_long",
    "always_short",
    "previous_target_persistence",
    f"{prereg.PRIMARY_POLICY_ID}_current_position_only",
    f"{prereg.PRIMARY_POLICY_ID}_masked_semantic_embedding",
    f"{prereg.PRIMARY_POLICY_ID}_metadata_frontmatter_only",
)


@dataclass(frozen=True)
class ReconstructedRewards:
    reward_tensor: np.ndarray
    terminal: np.ndarray
    reachable: np.ndarray


@dataclass(frozen=True)
class ResidualRewards:
    reward_tensor: np.ndarray
    deltas: np.ndarray


@dataclass(frozen=True)
class ResidualPolicyFamily:
    policies: OrderedDict[str, Any]
    feature_names: Mapping[str, str]
    fitted_q_count: int


def _utc_z(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or pd.isna(timestamp):
        raise ValueError("PSIM S5 schedule timestamp must be timezone aware")
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def reconstruct_reward_tensor(
    ledger: pd.DataFrame,
    source_rows: Sequence[Mapping[str, Any]],
) -> ReconstructedRewards:
    if (
        tuple(ledger.columns) != EXPECTED_LEDGER_COLUMNS
        or len(source_rows) != 366
        or ledger.duplicated(
            ["sequence_id", "current_position", "action_name"]
        ).any()
        or not ledger["reachable"].astype(bool).all()
        or set(ledger["current_position"]) - set(fqi.POSITION_NAMES)
        or set(ledger["action_name"]) - set(fqi.ACTION_NAMES)
    ):
        raise ValueError("PSIM S5 transition ledger schema changed")
    sequence_ids = [str(row["row_hash"]) for row in source_rows]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("PSIM S5 source sequence identity changed")
    sequence_index = {
        sequence_id: index
        for index, sequence_id in enumerate(sequence_ids)
    }
    if set(ledger["sequence_id"].astype(str)) != set(sequence_ids):
        raise ValueError("PSIM S5 ledger/source sequence alignment changed")
    count = len(source_rows)
    rewards = np.full((count, 3, 3), np.nan, dtype=np.float64)
    reachable = np.zeros((count, 3), dtype=bool)
    terminal = np.zeros(count, dtype=bool)
    observed_terminal: dict[int, bool] = {}
    observed_entry_time: dict[int, str] = {}
    for row in ledger.itertuples(index=False):
        state_index = sequence_index[str(row.sequence_id)]
        position_index = fqi.POSITION_INDEX[str(row.current_position)]
        action_index = fqi.ACTION_NAMES.index(str(row.action_name))
        reward = float(row.reward)
        if not np.isfinite(reward):
            raise ValueError("PSIM S5 transition reward is non-finite")
        if np.isfinite(rewards[state_index, position_index, action_index]):
            raise ValueError("PSIM S5 duplicate transition reward")
        rewards[state_index, position_index, action_index] = reward
        reachable[state_index, position_index] = True
        terminal_value = bool(row.terminal)
        if (
            state_index in observed_terminal
            and observed_terminal[state_index] != terminal_value
        ):
            raise ValueError("PSIM S5 terminal flag changed within state")
        observed_terminal[state_index] = terminal_value
        terminal[state_index] = terminal_value
        entry_time = _utc_z(row.entry_time)
        if (
            state_index in observed_entry_time
            and observed_entry_time[state_index] != entry_time
        ):
            raise ValueError("PSIM S5 entry clock changed within state")
        observed_entry_time[state_index] = entry_time
    if (
        int(reachable.sum()) != 1_096
        or int(np.isfinite(rewards).sum()) != 3_288
        or not np.array_equal(np.isfinite(rewards).all(axis=2), reachable)
        or int(terminal.sum()) != 1
        or not terminal[-1]
    ):
        raise RuntimeError("PSIM S5 reconstructed reward tensor changed")
    for index, source in enumerate(source_rows):
        if observed_entry_time.get(index) != _utc_z(source["decision_at"]):
            raise RuntimeError("PSIM S5 source/ledger entry clock changed")
    return ReconstructedRewards(
        reward_tensor=rewards,
        terminal=terminal,
        reachable=reachable,
    )


def direction_residualize_rewards(
    reward_tensor: np.ndarray,
    reachable: np.ndarray,
) -> ResidualRewards:
    rewards = np.asarray(reward_tensor, dtype=np.float64)
    mask = np.asarray(reachable, dtype=bool)
    if (
        rewards.ndim != 3
        or rewards.shape[1:] != (3, 3)
        or mask.shape != rewards.shape[:2]
        or not np.isfinite(rewards[mask]).all()
    ):
        raise ValueError("PSIM S5 reward tensor changed")
    long_index = fqi.ACTION_NAMES.index("TARGET_LONG")
    short_index = fqi.ACTION_NAMES.index("TARGET_SHORT")
    output = rewards.copy()
    deltas = np.zeros(3, dtype=np.float64)
    for position_index in range(3):
        selected = mask[:, position_index]
        differences = (
            rewards[selected, position_index, long_index]
            - rewards[selected, position_index, short_index]
        )
        if not len(differences) or not np.isfinite(differences).all():
            raise ValueError("PSIM S5 direction residual support changed")
        delta = 0.5 * float(np.mean(differences, dtype=np.float64))
        deltas[position_index] = delta
        output[selected, position_index, long_index] -= delta
        output[selected, position_index, short_index] += delta
        residual_difference = float(
            np.mean(
                output[selected, position_index, long_index]
                - output[selected, position_index, short_index],
                dtype=np.float64,
            )
        )
        if abs(residual_difference) > 1e-15:
            raise RuntimeError(
                "PSIM S5 direction residual mean invariant changed"
            )
    if (
        not np.array_equal(np.isnan(output), np.isnan(rewards))
        or not np.isfinite(output[mask]).all()
    ):
        raise RuntimeError("PSIM S5 residual reward finiteness changed")
    return ResidualRewards(reward_tensor=output, deltas=deltas)


def build_residual_ledger(
    ledger: pd.DataFrame,
    deltas: np.ndarray,
) -> pd.DataFrame:
    values = np.asarray(deltas, dtype=np.float64)
    if (
        tuple(ledger.columns) != EXPECTED_LEDGER_COLUMNS
        or values.shape != (3,)
        or not np.isfinite(values).all()
    ):
        raise ValueError("PSIM S5 residual ledger input changed")
    output = ledger.copy()
    applied: list[float] = []
    residual_rewards: list[float] = []
    for row in output.itertuples(index=False):
        position_index = fqi.POSITION_INDEX[str(row.current_position)]
        delta = float(values[position_index])
        if row.action_name == "TARGET_LONG":
            adjustment = -delta
        elif row.action_name == "TARGET_SHORT":
            adjustment = delta
        elif row.action_name == "TARGET_FLAT":
            adjustment = 0.0
        else:
            raise ValueError("PSIM S5 residual ledger action changed")
        applied.append(adjustment)
        residual_rewards.append(float(row.reward) + adjustment)
    insert_at = list(output.columns).index("reward") + 1
    output.insert(
        insert_at,
        "direction_residual_delta_applied",
        applied,
    )
    output.insert(
        insert_at + 1,
        "direction_residual_reward",
        residual_rewards,
    )
    output = output.loc[:, list(RESIDUAL_LEDGER_COLUMNS)]
    if (
        tuple(output.columns) != RESIDUAL_LEDGER_COLUMNS
        or len(output) != 3_288
        or not np.isfinite(output["direction_residual_reward"]).all()
    ):
        raise RuntimeError("PSIM S5 residual ledger output changed")
    return output


def fit_policy_family(
    feature_family: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    residual_rewards: np.ndarray,
    terminal: np.ndarray,
    reachable: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
) -> ResidualPolicyFamily:
    months = [str(row["decision_at"])[:7] for row in train_rows]
    policies: dict[str, Any] = {}
    feature_names: dict[str, str] = {}
    fitted_count = 0

    def fit_one(
        policy_id: str,
        feature_name: str,
        *,
        reward_mode: str = "base",
    ) -> Any:
        nonlocal fitted_count
        policy = fqi.fit_fitted_q(
            feature_family[feature_name][train_indices],
            residual_rewards,
            terminal,
            reachable,
            algorithm="ridge",
            reward_mode=reward_mode,
            months=months,
        )
        policies[policy_id] = policy
        feature_names[policy_id] = feature_name
        fitted_count += 1
        return policy

    primary = fit_one(prereg.PRIMARY_POLICY_ID, "semantic")
    permutation_id = f"{prereg.PRIMARY_POLICY_ID}_action_code_permutation"
    policies[permutation_id] = fqi.fit_action_code_permutation(
        feature_family["semantic"][train_indices],
        residual_rewards,
        terminal,
        reachable,
        algorithm="ridge",
        months=months,
    )
    feature_names[permutation_id] = "semantic"
    fitted_count += 1
    direction_id = f"{prereg.PRIMARY_POLICY_ID}_direction_flip"
    policies[direction_id] = fqi.DirectionFlipPolicy(primary)
    feature_names[direction_id] = "semantic"
    for suffix, reward_mode in (
        ("circular_21_reward", "circular_21"),
        ("within_month_shuffled_reward", "within_month_shuffled"),
    ):
        fit_one(
            f"{prereg.PRIMARY_POLICY_ID}_{suffix}",
            "semantic",
            reward_mode=reward_mode,
        )
    for suffix, feature_name in (
        ("current_position_only", "current_position_only"),
        ("masked_semantic_embedding", "masked_semantic_embedding"),
        ("metadata_frontmatter_only", "metadata_frontmatter_only"),
    ):
        fit_one(
            f"{prereg.PRIMARY_POLICY_ID}_{suffix}",
            feature_name,
        )
    ordered = OrderedDict(
        (policy_id, policies[policy_id])
        for policy_id in prereg.POLICY_FAMILY_IDS
    )
    if (
        tuple(ordered) != prereg.POLICY_FAMILY_IDS
        or fitted_count != 7
        or set(feature_names) != set(prereg.POLICY_FAMILY_IDS)
    ):
        raise RuntimeError("PSIM S5 fitted policy family changed")
    return ResidualPolicyFamily(
        policies=ordered,
        feature_names=feature_names,
        fitted_q_count=fitted_count,
    )


def build_schedule_family(
    family: ResidualPolicyFamily,
    feature_family: Mapping[str, np.ndarray],
    row_indices: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for policy_id in prereg.POLICY_FAMILY_IDS:
        feature_name = family.feature_names[policy_id]
        targets = fqi.rollout_policy(
            family.policies[policy_id],
            feature_family[feature_name][row_indices],
        )
        frames.append(
            pd.DataFrame(
                [
                    {
                        "policy_id": policy_id,
                        "sequence_id": str(rows[index]["row_hash"]),
                        "entry_time": _utc_z(rows[index]["decision_at"]),
                        "target": target,
                    }
                    for index, target in zip(row_indices, targets)
                ],
                columns=SCHEDULE_COLUMNS,
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    if (
        tuple(combined.columns) != SCHEDULE_COLUMNS
        or len(combined) != len(row_indices) * len(prereg.POLICY_FAMILY_IDS)
        or combined.duplicated(["policy_id", "sequence_id"]).any()
        or not combined["target"].isin(fqi.ACTION_NAMES).all()
        or tuple(combined["policy_id"].drop_duplicates())
        != prereg.POLICY_FAMILY_IDS
    ):
        raise RuntimeError("PSIM S5 schedule family changed")
    return combined


def build_delayed_primary_schedule(
    base_schedules: pd.DataFrame,
) -> pd.DataFrame:
    selected = base_schedules.loc[
        base_schedules["policy_id"].eq(prereg.PRIMARY_POLICY_ID)
    ].copy()
    selected["entry_time"] = [
        _utc_z(value)
        for value in (
            pd.to_datetime(selected["entry_time"], utc=True)
            + pd.Timedelta(minutes=5)
        )
    ]
    selected["sequence_id"] = (
        selected["sequence_id"].astype(str) + ":delay_5m"
    )
    selected = selected.reset_index(drop=True)
    if tuple(selected.columns) != SCHEDULE_COLUMNS or len(selected) != 365:
        raise RuntimeError("PSIM S5 delayed schedule changed")
    return selected


def _policy_targets(frame: pd.DataFrame, policy_id: str) -> pd.Series:
    selected = (
        frame.loc[frame["policy_id"].eq(policy_id)]
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    if len(selected) != 365:
        raise ValueError(f"PSIM S5 schedule length changed: {policy_id}")
    return selected["target"].astype(str)


def evaluate_schedule_readiness(
    schedules: pd.DataFrame,
    delayed: pd.DataFrame,
    s4_schedules: pd.DataFrame,
) -> dict[str, Any]:
    if (
        tuple(schedules.columns) != SCHEDULE_COLUMNS
        or tuple(delayed.columns) != SCHEDULE_COLUMNS
        or tuple(s4_schedules.columns) != SCHEDULE_COLUMNS
    ):
        raise ValueError("PSIM S5 readiness schedule columns changed")
    primary_targets = _policy_targets(schedules, prereg.PRIMARY_POLICY_ID)
    delayed_targets = _policy_targets(delayed, prereg.PRIMARY_POLICY_ID)
    permutation_id = f"{prereg.PRIMARY_POLICY_ID}_action_code_permutation"
    permutation_targets = _policy_targets(schedules, permutation_id)
    counts = {
        target: int((primary_targets == target).sum())
        for target in fqi.ACTION_NAMES
    }
    delayed_counts = {
        target: int((delayed_targets == target).sum())
        for target in fqi.ACTION_NAMES
    }
    nonflat = counts["TARGET_LONG"] + counts["TARGET_SHORT"]
    long_share = counts["TARGET_LONG"] / nonflat if nonflat else 0.0
    short_share = counts["TARGET_SHORT"] / nonflat if nonflat else 0.0
    degenerate_hamming: dict[str, int] = {}
    for control_id in DEGENERATE_CONTROL_IDS:
        source = (
            s4_schedules
            if control_id
            in {
                "always_flat",
                "always_long",
                "always_short",
                "previous_target_persistence",
            }
            else schedules
        )
        control_targets = _policy_targets(source, control_id)
        degenerate_hamming[control_id] = int(
            (primary_targets != control_targets).sum()
        )
    gates = {
        "base_primary_schedule_rows": len(primary_targets) == 365,
        "delayed_primary_schedule_rows": len(delayed_targets) == 365,
        "minimum_nonflat_target_rows": (
            nonflat >= prereg.MIN_NONFLAT_TARGET_ROWS
        ),
        "minimum_long_share": long_share >= prereg.MIN_DIRECTION_SHARE,
        "minimum_short_share": short_share >= prereg.MIN_DIRECTION_SHARE,
        "delayed_target_counts_equal_base": delayed_counts == counts,
        "action_code_permutation_exact_target_identity": (
            primary_targets.equals(permutation_targets)
        ),
        "all_degenerate_control_hamming_positive": all(
            value >= 1 for value in degenerate_hamming.values()
        ),
    }
    return {
        "primary_policy_id": prereg.PRIMARY_POLICY_ID,
        "target_counts": counts,
        "delayed_target_counts": delayed_counts,
        "nonflat_target_rows": nonflat,
        "long_share_of_nonflat_targets": long_share,
        "short_share_of_nonflat_targets": short_share,
        "action_code_permutation_policy_id": permutation_id,
        "action_code_permutation_mismatch_count": int(
            (primary_targets != permutation_targets).sum()
        ),
        "degenerate_control_target_hamming": degenerate_hamming,
        "gates": gates,
        "passed": all(gates.values()),
    }
