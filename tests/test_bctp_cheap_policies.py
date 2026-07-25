from __future__ import annotations

from collections import OrderedDict
import json

import numpy as np

from training import bctp_cheap_policies as p
from training import preregister_block_clearing_relational_topology as bcrt


def _state(i: int, *, unknown: bool = False) -> OrderedDict[str, object]:
    row: OrderedDict[str, object] = OrderedDict()
    row["entry_time"] = f"2020-01-{1 + i:02d}T00:00:00Z"
    for snap in ("s_minus_2", "s_minus_1", "s_0"):
        for j, (name, vocab) in enumerate(bcrt.TOKEN_SCHEMA):
            value = vocab[(i + j) % len(vocab)]
            row[f"{snap}__{name}"] = value
    if unknown:
        row["s_0__cadence_utilization"] = "NOT_IN_FROZEN_VOCAB"
    return row


def _states(n: int) -> list[OrderedDict[str, object]]:
    return [_state(i) for i in range(n)]


def _reward(n: int) -> np.ndarray:
    r = np.zeros((n, 3, 3), dtype=float)
    # Frozen action tensor order is FLAT, SHORT, LONG.
    r[:, :, 0] = 0.0
    r[:, :, 1] = -0.2
    r[:, :, 2] = 0.2
    return r


def test_schema_validation_and_unknown_fails_flat() -> None:
    states = _states(3)
    assert p.validate_source_state_schema(states) == states
    policy = p.fit_fitted_q(
        states,
        _reward(3),
        np.array([False, False, True]),
        np.ones((3, 3), dtype=bool),
        algorithm="linear",
    )
    assert p.rollout_policy(policy, [_state(99, unknown=True)]) == ["TARGET_FLAT"]


def test_bellman_continuation_and_terminal_zero_continuation() -> None:
    states = _states(2)
    rewards = np.zeros((2, 3, 3), dtype=float)
    # First row: flat action immediate utility 0.  Second row: being flat has
    # terminal immediate utility 1 for flat action.  Continuation should make
    # row-0 flat Q approximately gamma; terminal row must not add more gamma.
    rewards[1, 1, 0] = 1.0
    policy = p.fit_fitted_q(
        states,
        rewards,
        np.array([False, True]),
        np.array([[False, True, False], [False, True, False]]),
        algorithm="linear",
    )
    q0 = policy.predict_q(states[0], "POSITION_FLAT")
    q1 = policy.predict_q(states[1], "POSITION_FLAT")
    assert q0[0] > 0.98
    assert np.isclose(q1[0], 1.0, atol=1e-9)


def test_all_three_algorithms_and_deterministic_extra_trees() -> None:
    states = _states(8)
    terminal = np.zeros(8, dtype=bool)
    terminal[-1] = True
    reachable = np.ones((8, 3), dtype=bool)
    outputs = {}
    for algorithm in ("linear", "ridge", "extra_trees"):
        policy = p.fit_fitted_q(states, _reward(8), terminal, reachable, algorithm=algorithm)
        q = policy.predict_q(states[0], "POSITION_FLAT")
        assert q.shape == (3,)
        outputs[algorithm] = p.rollout_policy(policy, states)
    a = p.fit_fitted_q(states, _reward(8), terminal, reachable, algorithm="extra_trees")
    b = p.fit_fitted_q(states, _reward(8), terminal, reachable, algorithm="extra_trees")
    assert json.dumps(p.rollout_policy(a, states)) == json.dumps(p.rollout_policy(b, states))
    assert set(outputs) == {"linear", "ridge", "extra_trees"}


def test_feature_and_reward_controls_and_schedules() -> None:
    states = _states(30)
    terminal = np.zeros(30, dtype=bool)
    terminal[-1] = True
    reachable = np.ones((30, 3), dtype=bool)
    for feature_mode in ("full", "current_only", "reversed", "masked"):
        for reward_mode in ("base", "shuffled_month", "circular_21"):
            policy = p.fit_fitted_q(
                states,
                _reward(30),
                terminal,
                reachable,
                algorithm="ridge",
                feature_mode=feature_mode,
                reward_mode=reward_mode,
            )
            assert len(p.rollout_policy(policy, states)) == 30
    assert p.constant_schedule(states[:3], "TARGET_SHORT") == ["TARGET_SHORT"] * 3
    assert p.persistence_schedule(states[:3]) == ["TARGET_FLAT"] * 3


def test_exact_memory_unseen_flat_direction_flip_and_action_permutation_identity() -> None:
    states = _states(6)
    rewards = _reward(6)
    reachable = np.ones((6, 3), dtype=bool)
    memory = p.exact_signature_memory_policy(states, rewards, reachable)
    assert p.rollout_policy(memory, states[:1]) == ["TARGET_LONG"]
    assert p.rollout_policy(memory, [_state(99)]) == ["TARGET_FLAT"]

    base = p.fit_fitted_q(
        states,
        rewards,
        np.array([False, False, False, False, False, True]),
        reachable,
        algorithm="linear",
    )
    flipped = p.direction_flip(base)
    assert p.rollout_policy(flipped, states[:2]) == ["TARGET_SHORT", "TARGET_SHORT"]

    permuted = p.action_code_permutation(base)
    assert json.dumps(p.rollout_policy(permuted, states), separators=(",", ":")) == json.dumps(
        p.rollout_policy(base, states), separators=(",", ":")
    )
    fitted_permutation = p.action_code_permutation_policy(
        states,
        rewards,
        np.array([False, False, False, False, False, True]),
        reachable,
        algorithm="linear",
    )
    assert p.rollout_policy(fitted_permutation, states) == p.rollout_policy(
        base,
        states,
    )


def test_encoder_uses_frozen_vocabulary_not_observed_only() -> None:
    states = _states(2)
    policy = p.fit_fitted_q(
        states,
        _reward(2),
        np.array([False, True]),
        np.ones((2, 3), dtype=bool),
        algorithm="linear",
    )
    known_but_unobserved = _state(0)
    token_values = dict(bcrt.TOKEN_SCHEMA)["cadence_utilization"]
    known_but_unobserved["s_0__cadence_utilization"] = token_values[-1]
    q = policy.predict_q(known_but_unobserved, "POSITION_FLAT")
    assert np.isfinite(q).all()
