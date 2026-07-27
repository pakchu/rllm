from __future__ import annotations

import numpy as np
import pandas as pd

from training import (
    preregister_psim_d8_rllm2_s5_direction_residual_ridge_fqi as prereg,
)
from training import psim_direction_residual_fqi as residual
from training import psim_semantic_fqi_policies as fqi


def _source_rows(count: int = 366, *, year: int = 2020) -> list[dict]:
    start = pd.Timestamp(f"{year}-01-01T12:05:00Z")
    return [
        {
            "row_hash": f"{index:064x}",
            "decision_at": (
                start + pd.Timedelta(days=index)
            ).isoformat(),
        }
        for index in range(count)
    ]


def _ledger() -> pd.DataFrame:
    rows = _source_rows()
    output: list[dict] = []
    for state_index, source in enumerate(rows):
        positions = (
            ("POSITION_FLAT",)
            if state_index == 0
            else fqi.POSITION_NAMES
        )
        terminal = state_index == len(rows) - 1
        for position in positions:
            position_index = fqi.POSITION_INDEX[position]
            for action_index, action in enumerate(fqi.ACTION_NAMES):
                reward = (
                    0.001 * state_index
                    + 0.01 * position_index
                    + (-0.03, -0.02, 0.04)[action_index]
                )
                output.append(
                    {
                        "sequence_id": source["row_hash"],
                        "entry_time": source["decision_at"],
                        "current_position": position,
                        "action_name": action,
                        "action_target": fqi.ACTION_VALUES[action_index],
                        "executed_target": fqi.ACTION_VALUES[action_index],
                        "reachable": True,
                        "terminal": terminal,
                        "reward": reward,
                        "multiplier": np.exp(reward),
                        "held_path_downside_fraction": 0.0,
                        "changed_notional_fraction": 0.0,
                        "entry_cost": 0.0,
                        "terminal_cost": 0.0,
                        "funding_cash": 0.0,
                        "bars_held": 288,
                    }
                )
    return pd.DataFrame(output, columns=residual.EXPECTED_LEDGER_COLUMNS)


def test_reconstruct_and_direction_residualize_are_exact() -> None:
    ledger = _ledger()
    reconstructed = residual.reconstruct_reward_tensor(
        ledger,
        _source_rows(),
    )

    assert reconstructed.reward_tensor.shape == (366, 3, 3)
    assert reconstructed.reachable.shape == (366, 3)
    assert int(reconstructed.reachable.sum()) == 1_096
    assert int(np.isfinite(reconstructed.reward_tensor).sum()) == 3_288
    assert int(reconstructed.terminal.sum()) == 1
    assert reconstructed.terminal[-1]
    transformed = residual.direction_residualize_rewards(
        reconstructed.reward_tensor,
        reconstructed.reachable,
    )
    np.testing.assert_allclose(
        transformed.deltas,
        [0.03, 0.03, 0.03],
        rtol=0,
        atol=1e-15,
    )
    flat_index = fqi.ACTION_NAMES.index("TARGET_FLAT")
    np.testing.assert_array_equal(
        transformed.reward_tensor[:, :, flat_index],
        reconstructed.reward_tensor[:, :, flat_index],
    )
    long_index = fqi.ACTION_NAMES.index("TARGET_LONG")
    short_index = fqi.ACTION_NAMES.index("TARGET_SHORT")
    for position_index in range(3):
        selected = reconstructed.reachable[:, position_index]
        difference = np.mean(
            transformed.reward_tensor[selected, position_index, long_index]
            - transformed.reward_tensor[
                selected,
                position_index,
                short_index,
            ]
        )
        assert abs(float(difference)) <= 1e-15


def test_residual_ledger_preserves_original_and_applies_signed_delta() -> None:
    ledger = _ledger()
    transformed = residual.direction_residualize_rewards(
        residual.reconstruct_reward_tensor(
            ledger,
            _source_rows(),
        ).reward_tensor,
        residual.reconstruct_reward_tensor(
            ledger,
            _source_rows(),
        ).reachable,
    )
    output = residual.build_residual_ledger(
        ledger,
        transformed.deltas,
    )

    assert tuple(output.columns) == residual.RESIDUAL_LEDGER_COLUMNS
    assert len(output) == 3_288
    flat = output["action_name"].eq("TARGET_FLAT")
    short = output["action_name"].eq("TARGET_SHORT")
    long = output["action_name"].eq("TARGET_LONG")
    np.testing.assert_allclose(
        output.loc[flat, "direction_residual_delta_applied"],
        0.0,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        output.loc[short, "direction_residual_delta_applied"],
        0.03,
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        output.loc[long, "direction_residual_delta_applied"],
        -0.03,
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        output["direction_residual_reward"],
        output["reward"] + output["direction_residual_delta_applied"],
        rtol=0,
        atol=0,
    )


def test_complete_ridge_family_is_fixed_and_action_permutation_is_invariant() -> None:
    count = 40
    features = np.linspace(-1.0, 1.0, count * 32).reshape(count, 32)
    feature_family = {
        "semantic": features.astype(np.float32),
        "current_position_only": np.zeros((count, 0), dtype=np.float32),
        "masked_semantic_embedding": np.zeros(
            (count, 32),
            dtype=np.float32,
        ),
        "metadata_frontmatter_only": features[:, :11].astype(np.float32),
    }
    rewards = np.zeros((count, 3, 3), dtype=np.float64)
    rewards[:, :, 1] = features[:, 0, None]
    rewards[:, :, 2] = -features[:, 0, None]
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


def _schedule(
    policy_id: str,
    targets: list[str],
    *,
    delay_minutes: int = 0,
) -> pd.DataFrame:
    start = pd.Timestamp("2021-01-01T12:05:00Z") + pd.Timedelta(
        minutes=delay_minutes
    )
    return pd.DataFrame(
        [
            {
                "policy_id": policy_id,
                "sequence_id": (
                    f"{index:064x}"
                    + (":delay_5m" if delay_minutes else "")
                ),
                "entry_time": (
                    start + pd.Timedelta(days=index)
                ).isoformat().replace("+00:00", "Z"),
                "target": target,
            }
            for index, target in enumerate(targets)
        ],
        columns=residual.SCHEDULE_COLUMNS,
    )


def test_readiness_passes_only_balanced_active_distinct_schedule() -> None:
    primary_targets = (
        ["TARGET_LONG"] * 80
        + ["TARGET_SHORT"] * 80
        + ["TARGET_FLAT"] * 205
    )
    frames = [
        _schedule(prereg.PRIMARY_POLICY_ID, primary_targets),
        _schedule(
            f"{prereg.PRIMARY_POLICY_ID}_action_code_permutation",
            primary_targets,
        ),
        _schedule(
            f"{prereg.PRIMARY_POLICY_ID}_direction_flip",
            list(reversed(primary_targets)),
        ),
        _schedule(
            f"{prereg.PRIMARY_POLICY_ID}_circular_21_reward",
            primary_targets[21:] + primary_targets[:21],
        ),
        _schedule(
            f"{prereg.PRIMARY_POLICY_ID}_within_month_shuffled_reward",
            primary_targets[1:] + primary_targets[:1],
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
    ]
    schedules = pd.concat(frames, ignore_index=True)
    delayed = _schedule(
        prereg.PRIMARY_POLICY_ID,
        primary_targets,
        delay_minutes=5,
    )
    s4 = pd.concat(
        [
            _schedule("always_flat", ["TARGET_FLAT"] * 365),
            _schedule("always_long", ["TARGET_LONG"] * 365),
            _schedule("always_short", ["TARGET_SHORT"] * 365),
            _schedule(
                "previous_target_persistence",
                ["TARGET_FLAT"] * 365,
            ),
        ],
        ignore_index=True,
    )
    result = residual.evaluate_schedule_readiness(
        schedules,
        delayed,
        s4,
    )

    assert result["passed"] is True
    assert result["nonflat_target_rows"] == 160
    assert result["long_share_of_nonflat_targets"] == 0.5
    assert result["short_share_of_nonflat_targets"] == 0.5
    assert result["action_code_permutation_mismatch_count"] == 0
    assert all(
        value > 0
        for value in result["degenerate_control_target_hamming"].values()
    )


def test_readiness_rejects_direction_collapse_and_permutation_drift() -> None:
    primary_targets = ["TARGET_LONG"] * 100 + ["TARGET_FLAT"] * 265
    schedules = pd.concat(
        [
            _schedule(prereg.PRIMARY_POLICY_ID, primary_targets),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_action_code_permutation",
                ["TARGET_SHORT"] + primary_targets[1:],
            ),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_direction_flip",
                primary_targets,
            ),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_circular_21_reward",
                primary_targets,
            ),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_within_month_shuffled_reward",
                primary_targets,
            ),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_current_position_only",
                ["TARGET_FLAT"] * 365,
            ),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_masked_semantic_embedding",
                ["TARGET_FLAT"] * 365,
            ),
            _schedule(
                f"{prereg.PRIMARY_POLICY_ID}_metadata_frontmatter_only",
                ["TARGET_FLAT"] * 365,
            ),
        ],
        ignore_index=True,
    )
    delayed = _schedule(
        prereg.PRIMARY_POLICY_ID,
        primary_targets,
        delay_minutes=5,
    )
    s4 = pd.concat(
        [
            _schedule("always_flat", ["TARGET_FLAT"] * 365),
            _schedule("always_long", ["TARGET_LONG"] * 365),
            _schedule("always_short", ["TARGET_SHORT"] * 365),
            _schedule(
                "previous_target_persistence",
                ["TARGET_FLAT"] * 365,
            ),
        ],
        ignore_index=True,
    )
    result = residual.evaluate_schedule_readiness(
        schedules,
        delayed,
        s4,
    )

    assert result["passed"] is False
    assert result["gates"]["minimum_short_share"] is False
    assert (
        result["gates"]["action_code_permutation_exact_target_identity"]
        is False
    )
    assert result["action_code_permutation_mismatch_count"] == 1
