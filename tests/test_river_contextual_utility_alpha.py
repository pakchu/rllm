from dataclasses import replace

import numpy as np
import pandas as pd

from training.search_bidirectional_state_alpha import Config
from training.search_river_contextual_utility_alpha import (
    causal_gate_thresholds,
    contextual_utility_scores,
    effective_selection_signal_hash,
    executable_path_targets,
    policy_score_for_side,
    utility_policy_masks,
)


def test_executable_targets_use_delayed_open_and_held_bar_adverse_path():
    market = pd.DataFrame(
        {
            "open": [99.0, 100.0, 105.0, 110.0],
            "high": [100.0, 104.0, 112.0, 111.0],
            "low": [98.0, 98.0, 101.0, 109.0],
        }
    )
    cfg = replace(
        Config(input_csv="unused", output="unused"),
        leverage=0.5,
        fee_rate=0.0,
        slippage_rate=0.0,
    )

    targets = executable_path_targets(
        market,
        np.array([0]),
        cfg,
        hold_bars=2,
        entry_delay_bars=1,
    )

    np.testing.assert_allclose(targets["long_net"], [0.05])
    np.testing.assert_allclose(targets["short_net"], [-0.05])
    np.testing.assert_allclose(targets["long_mae_loss"], [0.01])
    np.testing.assert_allclose(targets["short_mae_loss"], [0.06])


def test_contextual_utility_penalizes_each_sides_own_mae():
    predictions = {
        "long_net": np.array([0.05]),
        "short_net": np.array([0.03]),
        "long_mae_loss": np.array([0.01]),
        "short_mae_loss": np.array([0.06]),
    }

    long_score, short_score, best = contextual_utility_scores(
        predictions, mae_penalty=1.0
    )

    np.testing.assert_allclose(long_score, [0.04])
    np.testing.assert_allclose(short_score, [-0.03])
    np.testing.assert_allclose(best, [0.04])


def test_causal_gate_excludes_current_score():
    scores = np.array([0.0, 1.0, 100.0, -1.0])
    changed_current = np.array([0.0, 1.0, -999.0, -1.0])

    threshold = causal_gate_thresholds(
        scores, rolling_window=3, quantile=0.75, min_periods=2
    )
    changed = causal_gate_thresholds(
        changed_current, rolling_window=3, quantile=0.75, min_periods=2
    )

    assert threshold[2] == 0.75
    assert changed[2] == threshold[2]
    assert threshold[3] == 50.5
    assert changed[3] != threshold[3]


def test_utility_policy_uses_flat_when_best_score_is_nonpositive():
    long_scores = np.array([0.02, -0.01, 0.03])
    short_scores = np.array([0.01, -0.02, 0.04])
    thresholds = np.array([0.0, 0.0, 0.0])
    positions = np.array([2, 5, 8])

    long_active, short_active = utility_policy_masks(
        long_scores,
        short_scores,
        thresholds,
        positions,
        12,
        side_policy="both",
    )

    np.testing.assert_array_equal(np.flatnonzero(long_active), [2])
    np.testing.assert_array_equal(np.flatnonzero(short_active), [8])


def test_side_only_policy_ignores_disallowed_sides_score():
    long_scores = np.array([0.02])
    short_scores = np.array([0.50])
    positions = np.array([2])

    long_active, short_active = utility_policy_masks(
        long_scores,
        short_scores,
        np.array([0.01]),
        positions,
        5,
        side_policy="long",
    )

    np.testing.assert_array_equal(np.flatnonzero(long_active), [2])
    assert not short_active.any()
    np.testing.assert_allclose(
        policy_score_for_side(long_scores, short_scores, side_policy="long"),
        long_scores,
    )


def test_selection_hash_deduplicates_signals_while_position_is_held():
    market = pd.DataFrame(
        {
            "open": np.full(10, 100.0),
            "high": np.full(10, 101.0),
            "low": np.full(10, 99.0),
        }
    )
    dates = pd.Series(pd.date_range("2023-01-01", periods=10, freq="5min"))
    sparse = np.zeros(10, dtype=bool)
    dense = np.zeros(10, dtype=bool)
    short = np.zeros(10, dtype=bool)
    sparse[0] = True
    dense[[0, 1, 2]] = True

    sparse_hash = effective_selection_signal_hash(
        market,
        dates,
        sparse,
        short,
        window=("2023-01-01", "2023-01-02"),
        hold_bars=2,
        stride_bars=1,
        minimum_signal_position=0,
    )
    dense_hash = effective_selection_signal_hash(
        market,
        dates,
        dense,
        short,
        window=("2023-01-01", "2023-01-02"),
        hold_bars=2,
        stride_bars=1,
        minimum_signal_position=0,
    )

    assert sparse_hash == dense_hash
