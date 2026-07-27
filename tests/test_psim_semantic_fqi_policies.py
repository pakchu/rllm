from __future__ import annotations

import numpy as np

from training import psim_semantic_fqi_policies as fqi


def _arrays(count: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.linspace(-1.0, 1.0, count).reshape(-1, 1)
    rewards = np.zeros((count, 3, 3), dtype=np.float64)
    rewards[:, :, 0] = 0.0
    rewards[:, :, 1] = features[:, 0, None]
    rewards[:, :, 2] = -features[:, 0, None]
    reachable = np.ones((count, 3), dtype=bool)
    reachable[0] = False
    reachable[0, fqi.POSITION_INDEX["POSITION_FLAT"]] = True
    rewards[~reachable] = np.nan
    terminal = np.zeros(count, dtype=bool)
    terminal[-1] = True
    return features, rewards, terminal, reachable


def test_ridge_fitted_q_uses_features_and_current_position() -> None:
    features, rewards, terminal, reachable = _arrays()
    policy = fqi.fit_fitted_q(
        features,
        rewards,
        terminal,
        reachable,
        algorithm="ridge",
    )
    targets = fqi.rollout_policy(policy, features)

    assert len(targets) == len(features)
    assert set(targets).issubset(set(fqi.ACTION_NAMES))
    assert any(target == "TARGET_LONG" for target in targets[:6])
    assert any(target == "TARGET_SHORT" for target in targets[6:])


def test_action_code_permutation_is_schedule_invariant_for_ridge() -> None:
    features, rewards, terminal, reachable = _arrays()
    primary = fqi.fit_fitted_q(
        features,
        rewards,
        terminal,
        reachable,
        algorithm="ridge",
    )
    permutation = fqi.fit_action_code_permutation(
        features,
        rewards,
        terminal,
        reachable,
        algorithm="ridge",
    )

    assert fqi.rollout_policy(primary, features) == fqi.rollout_policy(
        permutation,
        features,
    )


def test_reward_ablation_transforms_are_deterministic() -> None:
    features, rewards, terminal, reachable = _arrays(count=35)
    months = ["2020-01"] * 20 + ["2020-02"] * 15
    first = fqi.fit_fitted_q(
        features,
        rewards,
        terminal,
        reachable,
        algorithm="ridge",
        reward_mode="within_month_shuffled",
        months=months,
    )
    second = fqi.fit_fitted_q(
        features,
        rewards,
        terminal,
        reachable,
        algorithm="ridge",
        reward_mode="within_month_shuffled",
        months=months,
    )

    np.testing.assert_allclose(
        first.estimator.coefficient,
        second.estimator.coefficient,
        rtol=0,
        atol=0,
    )


def test_exact_payload_memory_fails_unknown_signatures_closed_to_flat() -> None:
    _, rewards, _, reachable = _arrays(count=4)
    signatures = ["a", "b", "c", "d"]
    table = fqi.fit_exact_payload_memory(signatures, rewards, reachable)

    targets = fqi.rollout_exact_payload_memory(table, ["unknown", "a"])

    assert targets[0] == "TARGET_FLAT"
    assert targets[1] in fqi.ACTION_NAMES
