"""All-action-mean residual reward and policy core for PSIM RLLM2 S6."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from training import (
    preregister_psim_d8_rllm2_s6_action_mean_residual_ridge_fqi as prereg,
)
from training import psim_direction_residual_fqi as s5_core
from training import psim_semantic_fqi_policies as fqi

SCHEDULE_COLUMNS = s5_core.SCHEDULE_COLUMNS
EXPECTED_LEDGER_COLUMNS = s5_core.EXPECTED_LEDGER_COLUMNS
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
    "action_mean_baseline",
    "position_grand_mean",
    "action_mean_residual_adjustment",
    "action_mean_residual_reward",
    "multiplier",
    "held_path_downside_fraction",
    "changed_notional_fraction",
    "entry_cost",
    "terminal_cost",
    "funding_cash",
    "bars_held",
)


@dataclass(frozen=True)
class ActionMeanResidualRewards:
    reward_tensor: np.ndarray
    action_means: np.ndarray
    position_grand_means: np.ndarray


@dataclass(frozen=True)
class ActionMeanPolicyFamily:
    policies: OrderedDict[str, Any]
    feature_names: Mapping[str, str]
    fitted_q_count: int


def action_mean_residualize_rewards(
    reward_tensor: np.ndarray,
    reachable: np.ndarray,
) -> ActionMeanResidualRewards:
    rewards = np.asarray(reward_tensor, dtype=np.float64)
    mask = np.asarray(reachable, dtype=bool)
    if (
        rewards.ndim != 3
        or rewards.shape[1:] != (3, 3)
        or mask.shape != rewards.shape[:2]
        or not np.isfinite(rewards[mask]).all()
        or not np.isnan(rewards[~mask]).all()
    ):
        raise ValueError("PSIM S6 reward tensor changed")
    output = rewards.copy()
    action_means = np.zeros((3, 3), dtype=np.float64)
    grand_means = np.zeros(3, dtype=np.float64)
    for position_index in range(3):
        selected = mask[:, position_index]
        if not selected.any():
            raise ValueError("PSIM S6 action-mean support changed")
        means = np.mean(
            rewards[selected, position_index, :],
            axis=0,
            dtype=np.float64,
        )
        grand = float(np.mean(means, dtype=np.float64))
        action_means[position_index] = means
        grand_means[position_index] = grand
        output[selected, position_index, :] = (
            rewards[selected, position_index, :] - means + grand
        )
        residual_means = np.mean(
            output[selected, position_index, :],
            axis=0,
            dtype=np.float64,
        )
        if float(np.max(residual_means) - np.min(residual_means)) > 1e-15:
            raise RuntimeError(
                "PSIM S6 action-mean residual invariant changed"
            )
    if (
        not np.array_equal(np.isnan(output), np.isnan(rewards))
        or not np.isfinite(output[mask]).all()
    ):
        raise RuntimeError("PSIM S6 residual reward finiteness changed")
    return ActionMeanResidualRewards(
        reward_tensor=output,
        action_means=action_means,
        position_grand_means=grand_means,
    )


def build_residual_ledger(
    ledger: pd.DataFrame,
    action_means: np.ndarray,
    position_grand_means: np.ndarray,
) -> pd.DataFrame:
    means = np.asarray(action_means, dtype=np.float64)
    grand = np.asarray(position_grand_means, dtype=np.float64)
    if (
        tuple(ledger.columns) != EXPECTED_LEDGER_COLUMNS
        or means.shape != (3, 3)
        or grand.shape != (3,)
        or not np.isfinite(means).all()
        or not np.isfinite(grand).all()
    ):
        raise ValueError("PSIM S6 residual ledger input changed")
    output = ledger.copy()
    baselines: list[float] = []
    grand_values: list[float] = []
    adjustments: list[float] = []
    residual_rewards: list[float] = []
    for row in output.itertuples(index=False):
        position_index = fqi.POSITION_INDEX[str(row.current_position)]
        action_index = fqi.ACTION_NAMES.index(str(row.action_name))
        baseline = float(means[position_index, action_index])
        position_grand = float(grand[position_index])
        adjustment = position_grand - baseline
        baselines.append(baseline)
        grand_values.append(position_grand)
        adjustments.append(adjustment)
        residual_rewards.append(float(row.reward) + adjustment)
    insert_at = list(output.columns).index("reward") + 1
    for offset, (name, values) in enumerate(
        (
            ("action_mean_baseline", baselines),
            ("position_grand_mean", grand_values),
            ("action_mean_residual_adjustment", adjustments),
            ("action_mean_residual_reward", residual_rewards),
        )
    ):
        output.insert(insert_at + offset, name, values)
    output = output.loc[:, list(RESIDUAL_LEDGER_COLUMNS)]
    if (
        tuple(output.columns) != RESIDUAL_LEDGER_COLUMNS
        or len(output) != 3_288
        or not np.isfinite(output["action_mean_residual_reward"]).all()
    ):
        raise RuntimeError("PSIM S6 residual ledger output changed")
    return output


def fit_policy_family(
    feature_family: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    residual_rewards: np.ndarray,
    terminal: np.ndarray,
    reachable: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
) -> ActionMeanPolicyFamily:
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
        raise RuntimeError("PSIM S6 fitted policy family changed")
    return ActionMeanPolicyFamily(
        policies=ordered,
        feature_names=feature_names,
        fitted_q_count=fitted_count,
    )


def build_schedule_family(
    family: ActionMeanPolicyFamily,
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
                        "entry_time": s5_core._utc_z(
                            rows[index]["decision_at"]
                        ),
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
        raise RuntimeError("PSIM S6 schedule family changed")
    return combined


def build_delayed_primary_schedule(
    base_schedules: pd.DataFrame,
) -> pd.DataFrame:
    selected = base_schedules.loc[
        base_schedules["policy_id"].eq(prereg.PRIMARY_POLICY_ID)
    ].copy()
    selected["entry_time"] = [
        s5_core._utc_z(value)
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
        raise RuntimeError("PSIM S6 delayed schedule changed")
    return selected


def _policy_schedule(frame: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    selected = (
        frame.loc[frame["policy_id"].eq(policy_id)]
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    if (
        len(selected) != 365
        or selected["sequence_id"].duplicated().any()
        or selected["entry_time"].duplicated().any()
    ):
        raise ValueError(f"PSIM S6 schedule length changed: {policy_id}")
    return selected


def _policy_targets(frame: pd.DataFrame, policy_id: str) -> pd.Series:
    return _policy_schedule(frame, policy_id)["target"].astype(str)


def evaluate_schedule_readiness(
    schedules: pd.DataFrame,
    delayed: pd.DataFrame,
    s4_schedules: pd.DataFrame,
    s5_schedules: pd.DataFrame,
) -> dict[str, Any]:
    if any(
        tuple(frame.columns) != SCHEDULE_COLUMNS
        for frame in (schedules, delayed, s4_schedules, s5_schedules)
    ):
        raise ValueError("PSIM S6 readiness schedule columns changed")
    valid_action_targets = all(
        frame["target"].isin(fqi.ACTION_NAMES).all()
        for frame in (schedules, delayed, s4_schedules, s5_schedules)
    )
    primary_schedule = _policy_schedule(
        schedules,
        prereg.PRIMARY_POLICY_ID,
    )
    delayed_schedule = _policy_schedule(
        delayed,
        prereg.PRIMARY_POLICY_ID,
    )
    primary_targets = primary_schedule["target"].astype(str)
    delayed_targets = delayed_schedule["target"].astype(str)
    base_entry_times = pd.to_datetime(
        primary_schedule["entry_time"],
        utc=True,
    )
    delayed_entry_times = pd.to_datetime(
        delayed_schedule["entry_time"],
        utc=True,
    )
    delayed_sequence_identity = delayed_schedule["sequence_id"].tolist() == [
        f"{value}:delay_5m"
        for value in primary_schedule["sequence_id"].astype(str)
    ]
    delayed_entry_time_identity = bool(
        (
            delayed_entry_times
            == base_entry_times + pd.Timedelta(minutes=5)
        ).all()
    )
    delayed_target_identity = primary_targets.equals(delayed_targets)
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
    for control_id in prereg.DEGENERATE_CONTROL_IDS:
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
    s5_targets = _policy_targets(s5_schedules, s5_core.prereg.PRIMARY_POLICY_ID)
    s5_hamming = int((primary_targets != s5_targets).sum())
    gates = {
        "all_schedule_targets_in_action_domain": valid_action_targets,
        "base_primary_schedule_rows": len(primary_targets) == 365,
        "delayed_primary_schedule_rows": len(delayed_targets) == 365,
        "minimum_nonflat_target_rows": (
            nonflat >= prereg.MIN_NONFLAT_TARGET_ROWS
        ),
        "minimum_long_share": long_share >= prereg.MIN_DIRECTION_SHARE,
        "minimum_short_share": short_share >= prereg.MIN_DIRECTION_SHARE,
        "delayed_target_counts_equal_base": delayed_counts == counts,
        "delayed_schedule_exact_5m_identity": (
            delayed_sequence_identity
            and delayed_entry_time_identity
            and delayed_target_identity
        ),
        "action_code_permutation_exact_target_identity": (
            primary_targets.equals(permutation_targets)
        ),
        "all_degenerate_control_hamming_positive": all(
            value >= 1 for value in degenerate_hamming.values()
        ),
        "minimum_hamming_distance_from_s5_primary": s5_hamming >= 1,
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
        "s5_primary_policy_id": s5_core.prereg.PRIMARY_POLICY_ID,
        "s5_primary_target_hamming": s5_hamming,
        "gates": gates,
        "passed": all(gates.values()),
    }
