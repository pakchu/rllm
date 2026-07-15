from __future__ import annotations

import numpy as np
import pandas as pd

from training import select_ccnear_primaryweak_ridge_pre2024 as selector


def test_candidate_is_single_fixed_three_feature_ridge() -> None:
    assert selector.FEATURE_NAMES == (
        "cash_vote_36",
        "derivative_vote_36",
        "refill_vote_36",
    )
    assert selector.LOOKBACK_BARS == 36
    assert selector.RIDGE_ALPHA == 1.0
    assert selector.HOLD_BARS == 288


def test_recent_side_is_same_or_past_only() -> None:
    schedule = pd.DataFrame(
        {
            "signal_position": [2, 8],
            "side": [1, -1],
        }
    )
    values = selector.recent_side(schedule, frame_length=12, lookback_bars=3)
    assert values.tolist() == [0, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1]


def test_vote_matrix_compresses_primary_families_without_future_rows() -> None:
    schedules = {
        family: pd.DataFrame({"signal_position": [1], "side": [1]})
        for family in (*selector.CASH_FAMILIES, *selector.DERIVATIVE_FAMILIES, "rift")
    }
    matrix = selector.build_vote_matrix(schedules, frame_length=3)
    assert matrix.shape == (3, 3)
    assert np.array_equal(matrix[0], [0.0, 0.0, 0.0])
    assert np.array_equal(matrix[1], [3.0, 2.0, 1.0])


def test_all_declared_windows_end_before_2024() -> None:
    assert max(pd.Timestamp(end) for _, end in selector.WINDOWS.values()) == pd.Timestamp(
        "2024-01-01"
    )
