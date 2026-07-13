from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_spot_perp_phase_slip_alpha import (
    INVALID,
    phase_signals,
    symbolic_state,
)


def test_symbolic_state_prefix_does_not_depend_on_future_suffix() -> None:
    prefix = pd.Series(np.sin(np.arange(420, dtype=float) / 13.0) / 100.0)
    full = pd.concat([prefix, pd.Series([100.0] * 30)], ignore_index=True)

    np.testing.assert_array_equal(
        symbolic_state(full)[: len(prefix)],
        symbolic_state(prefix),
    )


def test_phase_signal_uses_disjoint_prior_lock_slip_and_current_relock() -> None:
    leader = np.full(12, INVALID, dtype=np.int8)
    follower = np.full(12, INVALID, dtype=np.int8)
    leader[2:5] = [1, -1, 1]
    follower[2:5] = [1, -1, 1]
    leader[5:7] = [1, 1]
    follower[5:7] = [0, 0]
    leader[7] = 1
    follower[7] = 1

    long_signal, short_signal, diagnostics = phase_signals(
        leader,
        follower,
        lock_window=3,
        slip_bars=2,
        min_excess=2,
        relock_mode="hard_relock",
    )

    assert np.flatnonzero(long_signal).tolist() == [7]
    assert not short_signal.any()
    assert diagnostics["prior_lock"][7]
    assert diagnostics["relock"][7]


def test_phase_signal_does_not_depend_on_future_suffix_and_flip_is_exact() -> None:
    leader = np.full(12, INVALID, dtype=np.int8)
    follower = np.full(12, INVALID, dtype=np.int8)
    leader[2:8] = [1, 1, 1, -1, -1, -1]
    follower[2:8] = [1, 1, 1, 0, 0, -1]
    kwargs = {
        "lock_window": 3,
        "slip_bars": 2,
        "min_excess": 2,
        "relock_mode": "hard_relock",
    }

    long_signal, short_signal, _ = phase_signals(leader, follower, **kwargs)
    changed_leader = leader.copy()
    changed_follower = follower.copy()
    changed_leader[8:] = 1
    changed_follower[8:] = -1
    changed_long, changed_short, _ = phase_signals(changed_leader, changed_follower, **kwargs)
    np.testing.assert_array_equal(changed_long[:8], long_signal[:8])
    np.testing.assert_array_equal(changed_short[:8], short_signal[:8])

    flip_long, flip_short, _ = phase_signals(leader, follower, **kwargs, flip=True)
    np.testing.assert_array_equal(flip_long, short_signal)
    np.testing.assert_array_equal(flip_short, long_signal)


def test_relock_requires_follower_to_newly_enter_leader_direction() -> None:
    leader = np.full(12, INVALID, dtype=np.int8)
    follower = np.full(12, INVALID, dtype=np.int8)
    leader[2:5] = follower[2:5] = 1
    leader[5:7] = [1, 1]
    follower[5:7] = [0, 1]
    leader[7] = follower[7] = 1

    long_signal, short_signal, _ = phase_signals(
        leader,
        follower,
        lock_window=3,
        slip_bars=2,
        min_excess=1,
        relock_mode="hard_relock",
    )

    assert not long_signal[7]
    assert not short_signal[7]
