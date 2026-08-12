import numpy as np

from training.build_high_volatility_equal_variance_clock_concordance_relay_support import (
    equal_variance_segments,
    three_of_four_side,
    unanimous_side,
)


def test_equal_variance_segments_are_ordered_and_complete() -> None:
    returns = np.full(95, 0.01)
    segments, counts = equal_variance_segments(returns)
    assert counts.tolist() == [24, 24, 24, 23]
    assert np.isclose(segments.sum(), returns.sum())
    assert unanimous_side(segments) == 1


def test_large_return_can_leave_intrinsic_quartile_empty() -> None:
    returns = np.full(95, 0.001)
    returns[1] = 1.0
    _, counts = equal_variance_segments(returns)
    assert (counts == 0).any()


def test_concordance_helpers_are_strict() -> None:
    assert unanimous_side(np.array([-1.0, -2.0, -3.0, -4.0])) == -1
    assert unanimous_side(np.array([1.0, 2.0, -3.0, 4.0])) == 0
    assert three_of_four_side(np.array([1.0, 2.0, -3.0, 4.0])) == 1
