from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import (
    preregister_psim_d8_rllm2_s6_action_mean_residual_ridge_fqi as prereg,
)
from training import psim_action_mean_residual_fqi as residual
from training import psim_semantic_fqi_policies as fqi


def test_action_mean_residual_equalizes_all_action_means() -> None:
    count = 12
    rewards = np.zeros((count, 3, 3), dtype=np.float64)
    state = np.linspace(-0.2, 0.3, count)
    for position in range(3):
        rewards[:, position, 0] = state + position * 0.01
        rewards[:, position, 1] = 2 * state - 0.04 + position * 0.02
        rewards[:, position, 2] = -state + 0.08 - position * 0.01
    reachable = np.ones((count, 3), dtype=bool)
    transformed = residual.action_mean_residualize_rewards(
        rewards,
        reachable,
    )

    assert transformed.action_means.shape == (3, 3)
    assert transformed.position_grand_means.shape == (3,)
    for position in range(3):
        expected_means = rewards[:, position, :].mean(axis=0)
        expected_grand = expected_means.mean()
        np.testing.assert_allclose(
            transformed.action_means[position],
            expected_means,
            rtol=0,
            atol=0,
        )
        assert transformed.position_grand_means[position] == expected_grand
        observed = transformed.reward_tensor[:, position, :].mean(axis=0)
        np.testing.assert_allclose(
            observed,
            np.repeat(expected_grand, 3),
            rtol=0,
            atol=1e-15,
        )


@pytest.mark.parametrize(
    ("finite_reward", "reachable"),
    (
        (True, False),
        (False, True),
    ),
)
def test_action_mean_residual_rejects_support_mismatch(
    finite_reward: bool,
    reachable: bool,
) -> None:
    rewards = np.zeros((4, 3, 3), dtype=np.float64)
    mask = np.ones((4, 3), dtype=bool)
    mask[0, 0] = reachable
    if not finite_reward:
        rewards[0, 0, :] = np.nan

    with pytest.raises(ValueError, match="reward tensor changed"):
        residual.action_mean_residualize_rewards(rewards, mask)


def test_action_mean_residual_rejects_partial_unreachable_reward() -> None:
    rewards = np.zeros((4, 3, 3), dtype=np.float64)
    mask = np.ones((4, 3), dtype=bool)
    mask[0, 0] = False
    rewards[0, 0, :] = [np.nan, 0.0, np.nan]

    with pytest.raises(ValueError, match="reward tensor changed"):
        residual.action_mean_residualize_rewards(rewards, mask)


def test_residual_ledger_preserves_original_and_records_baselines() -> None:
    rows: list[dict] = []
    for state in range(366):
        positions = ("POSITION_FLAT",) if state == 0 else fqi.POSITION_NAMES
        for position in positions:
            for action in fqi.ACTION_NAMES:
                rows.append(
                    {
                        "sequence_id": f"{state:064x}",
                        "entry_time": (
                            pd.Timestamp("2020-01-01T12:05:00Z")
                            + pd.Timedelta(days=state)
                        ).isoformat(),
                        "current_position": position,
                        "action_name": action,
                        "action_target": 0.0,
                        "executed_target": 0.0,
                        "reachable": True,
                        "terminal": state == 365,
                        "reward": float(state) / 10_000,
                        "multiplier": 1.0,
                        "held_path_downside_fraction": 0.0,
                        "changed_notional_fraction": 0.0,
                        "entry_cost": 0.0,
                        "terminal_cost": 0.0,
                        "funding_cash": 0.0,
                        "bars_held": 288,
                    }
                )
    ledger = pd.DataFrame(rows, columns=residual.EXPECTED_LEDGER_COLUMNS)
    means = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float64,
    )
    grand = means.mean(axis=1)
    output = residual.build_residual_ledger(ledger, means, grand)

    assert tuple(output.columns) == residual.RESIDUAL_LEDGER_COLUMNS
    assert len(output) == 3_288
    np.testing.assert_allclose(
        output["action_mean_residual_reward"],
        output["reward"] + output["action_mean_residual_adjustment"],
        rtol=0,
        atol=0,
    )
    first = output.iloc[0]
    position_index = fqi.POSITION_INDEX[first["current_position"]]
    action_index = fqi.ACTION_NAMES.index(first["action_name"])
    assert first["action_mean_baseline"] == means[
        position_index,
        action_index,
    ]
    assert first["position_grand_mean"] == grand[position_index]


def test_complete_family_has_seven_ridge_fits_and_invariant_codes() -> None:
    count = 40
    semantic = np.linspace(-1.0, 1.0, count * 32).reshape(count, 32)
    feature_family = {
        "semantic": semantic.astype(np.float32),
        "current_position_only": np.zeros((count, 0), dtype=np.float32),
        "masked_semantic_embedding": np.zeros(
            (count, 32),
            dtype=np.float32,
        ),
        "metadata_frontmatter_only": semantic[:, :11].astype(np.float32),
    }
    rewards = np.zeros((count, 3, 3), dtype=np.float64)
    rewards[:, :, 1] = semantic[:, 0, None]
    rewards[:, :, 2] = -semantic[:, 0, None]
    reachable = np.ones((count, 3), dtype=bool)
    terminal = np.zeros(count, dtype=bool)
    terminal[-1] = True
    rows = [
        {
            "decision_at": (
                pd.Timestamp("2020-01-01T12:05:00Z")
                + pd.Timedelta(days=index)
            ).isoformat()
        }
        for index in range(count)
    ]
    family = residual.fit_policy_family(
        feature_family,
        np.arange(count),
        rewards,
        terminal,
        reachable,
        rows,
    )

    assert tuple(family.policies) == prereg.POLICY_FAMILY_IDS
    assert family.fitted_q_count == 7
    primary = fqi.rollout_policy(
        family.policies[prereg.PRIMARY_POLICY_ID],
        feature_family["semantic"],
    )
    permutation = fqi.rollout_policy(
        family.policies[
            f"{prereg.PRIMARY_POLICY_ID}_action_code_permutation"
        ],
        feature_family["semantic"],
    )
    assert primary == permutation


def _schedule(policy_id: str, targets: list[str], delay: int = 0) -> pd.DataFrame:
    start = pd.Timestamp("2021-01-01T12:05:00Z") + pd.Timedelta(minutes=delay)
    return pd.DataFrame(
        [
            {
                "policy_id": policy_id,
                "sequence_id": f"{index:064x}" + (":delay_5m" if delay else ""),
                "entry_time": (
                    start + pd.Timedelta(days=index)
                ).isoformat().replace("+00:00", "Z"),
                "target": target,
            }
            for index, target in enumerate(targets)
        ],
        columns=residual.SCHEDULE_COLUMNS,
    )


def test_readiness_includes_unchanged_gates_and_s5_hamming() -> None:
    targets = (
        ["TARGET_LONG"] * 90
        + ["TARGET_SHORT"] * 90
        + ["TARGET_FLAT"] * 185
    )
    schedules = pd.concat(
        [
            _schedule(prereg.PRIMARY_POLICY_ID, targets),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_action_code_permutation",
                targets,
            ),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_direction_flip",
                list(reversed(targets)),
            ),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_circular_21_reward",
                targets[21:] + targets[:21],
            ),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_within_month_shuffled_reward",
                targets[1:] + targets[:1],
            ),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_current_position_only",
                ["TARGET_FLAT"] * 365,
            ),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_masked_semantic_embedding",
                ["TARGET_LONG"] * 365,
            ),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_metadata_frontmatter_only",
                ["TARGET_SHORT"] * 365,
            ),
        ],
        ignore_index=True,
    )
    s4 = pd.concat(
        [
            _schedule("always_flat", ["TARGET_FLAT"] * 365),
            _schedule("always_long", ["TARGET_LONG"] * 365),
            _schedule("always_short", ["TARGET_SHORT"] * 365),
            _schedule("previous_target_persistence", ["TARGET_FLAT"] * 365),
        ],
        ignore_index=True,
    )
    s5 = _schedule(
        residual.s5_core.prereg.PRIMARY_POLICY_ID,
        ["TARGET_FLAT"] * 365,
    )
    result = residual.evaluate_schedule_readiness(
        schedules,
        _schedule(prereg.PRIMARY_POLICY_ID, targets, delay=5),
        s4,
        s5,
    )

    assert result["passed"] is True
    assert result["nonflat_target_rows"] == 180
    assert result["long_share_of_nonflat_targets"] == 0.5
    assert result["short_share_of_nonflat_targets"] == 0.5
    assert result["action_code_permutation_mismatch_count"] == 0
    assert result["s5_primary_target_hamming"] == 180
    assert all(result["gates"].values())


def test_readiness_rejects_invalid_action_target() -> None:
    targets = (
        ["TARGET_LONG"] * 90
        + ["TARGET_SHORT"] * 90
        + ["TARGET_BOGUS"] * 185
    )
    schedules = pd.concat(
        [
            _schedule(policy_id, targets)
            for policy_id in prereg.POLICY_FAMILY_IDS
        ],
        ignore_index=True,
    )
    s4 = pd.concat(
        [
            _schedule("always_flat", ["TARGET_FLAT"] * 365),
            _schedule("always_long", ["TARGET_LONG"] * 365),
            _schedule("always_short", ["TARGET_SHORT"] * 365),
            _schedule("previous_target_persistence", ["TARGET_FLAT"] * 365),
        ],
        ignore_index=True,
    )
    s5 = _schedule(
        residual.s5_core.prereg.PRIMARY_POLICY_ID,
        ["TARGET_FLAT"] * 365,
    )
    result = residual.evaluate_schedule_readiness(
        schedules,
        _schedule(prereg.PRIMARY_POLICY_ID, targets, delay=5),
        s4,
        s5,
    )

    assert result["passed"] is False
    assert result["gates"]["all_schedule_targets_in_action_domain"] is False


def test_readiness_rejects_schedule_without_exact_delay() -> None:
    targets = (
        ["TARGET_LONG"] * 90
        + ["TARGET_SHORT"] * 90
        + ["TARGET_FLAT"] * 185
    )
    schedules = pd.concat(
        [
            _schedule(policy_id, targets)
            for policy_id in prereg.POLICY_FAMILY_IDS
        ],
        ignore_index=True,
    )
    s4 = pd.concat(
        [
            _schedule("always_flat", ["TARGET_FLAT"] * 365),
            _schedule("always_long", ["TARGET_LONG"] * 365),
            _schedule("always_short", ["TARGET_SHORT"] * 365),
            _schedule("previous_target_persistence", ["TARGET_FLAT"] * 365),
        ],
        ignore_index=True,
    )
    s5 = _schedule(
        residual.s5_core.prereg.PRIMARY_POLICY_ID,
        ["TARGET_FLAT"] * 365,
    )
    result = residual.evaluate_schedule_readiness(
        schedules,
        _schedule(prereg.PRIMARY_POLICY_ID, targets),
        s4,
        s5,
    )

    assert result["passed"] is False
    assert result["gates"]["delayed_schedule_exact_5m_identity"] is False
