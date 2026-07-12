import numpy as np
import pandas as pd

from training.search_tabicl_foundation_alpha import (
    HOLD_BARS,
    policy_masks,
    split_mask_for_anchors,
    top10_promotions,
)


def test_anchor_split_purges_label_crossing_boundary():
    dates = pd.Series(pd.date_range("2022-12-30", periods=HOLD_BARS + 10, freq="5min"))
    positions = np.array([0, 5])

    mask = split_mask_for_anchors(dates, positions, "2022-01-01", "2023-01-01")

    assert not mask.any()


def test_policy_masks_enter_only_at_anchor_positions():
    scores = np.array([-1.0, 0.0, 1.0])
    positions = np.array([2, 5, 8])
    long_active, short_active = policy_masks(
        scores,
        positions,
        12,
        side_policy="both",
        low_threshold=-0.5,
        high_threshold=0.5,
    )

    np.testing.assert_array_equal(np.flatnonzero(long_active), [8])
    np.testing.assert_array_equal(np.flatnonzero(short_active), [2])


def test_top10_promotions_excludes_rank11():
    loser = {"passes_alpha_pool": False, "passes_live_grade": False}
    winner = {"passes_alpha_pool": True, "passes_live_grade": True}
    selected = [loser, winner, *([loser] * 8), winner]

    assert top10_promotions(selected) == ([winner], [winner])
