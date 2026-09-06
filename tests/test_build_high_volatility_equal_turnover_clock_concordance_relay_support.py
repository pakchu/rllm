import numpy as np
import pandas as pd

from training import build_high_volatility_equal_turnover_clock_concordance_relay_support as support


def test_prior_rank_excludes_current() -> None:
    ranks = support.prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_equal_turnover_assignment_uses_cumulative_before_minute() -> None:
    turnover = np.ones(480)
    returns = np.arange(480, dtype=float)
    segment_returns, counts = support.turnover_segments(turnover, returns)
    assert counts.tolist() == [120, 120, 120, 120]
    assert segment_returns.tolist() == [
        returns[:120].sum(),
        returns[120:240].sum(),
        returns[240:360].sum(),
        returns[360:].sum(),
    ]


def test_indivisible_large_minute_can_leave_segment_empty() -> None:
    turnover = np.ones(480)
    turnover[0] = 10_000
    returns = np.ones(480)
    _, counts = support.turnover_segments(turnover, returns)
    assert counts[0] == 1
    assert counts[1] == 0
    assert counts[2] == 0
    assert counts[3] == 479


def test_common_side_is_strict_and_three_of_four_is_control_only() -> None:
    assert support.common_side(np.array([1.0, 2.0, 3.0, 4.0])) == 1
    assert support.common_side(np.array([-1.0, -2.0, -3.0, -4.0])) == -1
    assert support.common_side(np.array([1.0, 2.0, 3.0, -4.0])) == 0
    assert support.common_side(np.array([1.0, 2.0, 3.0, -4.0]), 3) == 1
    assert support.common_side(np.array([1.0, 2.0, 0.0, 4.0]), 3) == 0


def test_onset_compares_previous_source_valid_decision() -> None:
    source_valid = pd.Series([True, False, True, True, True])
    eligible = pd.Series([False, False, True, True, False])
    onset = support.onset_after_previous_source_valid(source_valid, eligible)
    assert onset.tolist() == [False, False, True, False, False]


def test_contract_is_frozen() -> None:
    assert support.PREREG_SHA == "64a9e6099c1493ca8168928cde06f390cb9bed3cc609e0874568f6a1494e6bf2"
    assert support.CONTROLS == (
        "no_variation_gate",
        "three_of_four_concordance",
        "equal_physical_time",
        "one_decision_stale_geometry",
        "direction_flip",
        "same_clock_forced_long",
    )
    assert "quote_asset_volume" in support.QUERY
    assert "taker_buy" not in support.QUERY
