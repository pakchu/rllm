from __future__ import annotations

import numpy as np

from training.search_orderflow_trophic_chirp_alpha import chirp_signals


def _events(length: int, long_at: tuple[int, ...], short_at: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    long_events = np.zeros(length, dtype=bool)
    short_events = np.zeros(length, dtype=bool)
    long_events[list(long_at)] = True
    short_events[list(short_at)] = True
    return long_events, short_events


def test_acceleration_continues_on_compressing_same_direction_gaps() -> None:
    long_events, short_events = _events(20, (1, 9, 13))

    long_active, short_active, diagnostics = chirp_signals(
        long_events,
        short_events,
        max_gap_bars=12,
        branch="acceleration_continuation",
    )

    np.testing.assert_array_equal(np.flatnonzero(long_active), [13])
    assert not short_active.any()
    assert diagnostics["gap_ratio"][13] == 0.5


def test_deceleration_reverses_on_expanding_same_direction_gaps() -> None:
    long_events, short_events = _events(20, (1, 5, 13))

    long_active, short_active, diagnostics = chirp_signals(
        long_events,
        short_events,
        max_gap_bars=12,
        branch="deceleration_reversal",
    )

    assert not long_active.any()
    np.testing.assert_array_equal(np.flatnonzero(short_active), [13])
    assert diagnostics["gap_ratio"][13] == 2.0


def test_opposite_event_quarantines_triplet_unless_control_removes_it() -> None:
    long_events, short_events = _events(20, (1, 5, 13), (3,))
    kwargs = {"max_gap_bars": 12, "branch": "deceleration_reversal"}

    clean_long, clean_short, diagnostics = chirp_signals(long_events, short_events, **kwargs)
    loose_long, loose_short, _ = chirp_signals(
        long_events,
        short_events,
        require_clean_triplet=False,
        **kwargs,
    )

    assert not clean_long.any() and not clean_short.any()
    assert not diagnostics["clean_triplet"][13]
    assert not loose_long.any()
    np.testing.assert_array_equal(np.flatnonzero(loose_short), [13])


def test_max_gap_and_direction_flip_are_exact() -> None:
    long_events, short_events = _events(30, (1, 11, 16))
    kwargs = {"max_gap_bars": 12, "branch": "acceleration_continuation"}

    base_long, base_short, _ = chirp_signals(long_events, short_events, **kwargs)
    flip_long, flip_short, _ = chirp_signals(long_events, short_events, flip=True, **kwargs)
    too_short_long, too_short_short, _ = chirp_signals(
        long_events,
        short_events,
        max_gap_bars=8,
        branch="acceleration_continuation",
    )

    np.testing.assert_array_equal(base_long, flip_short)
    np.testing.assert_array_equal(base_short, flip_long)
    assert not too_short_long.any() and not too_short_short.any()


def test_chirp_prefix_is_future_suffix_independent() -> None:
    prefix_long, prefix_short = _events(30, (1, 9, 13, 25), (17, 20, 22))
    suffix_long, suffix_short = _events(10, (0, 1, 2), (5, 7, 8))
    full_long = np.r_[prefix_long, suffix_long]
    full_short = np.r_[prefix_short, suffix_short]
    kwargs = {"max_gap_bars": 16, "branch": "acceleration_continuation"}

    expected_long, expected_short, expected_diagnostics = chirp_signals(prefix_long, prefix_short, **kwargs)
    actual_long, actual_short, actual_diagnostics = chirp_signals(full_long, full_short, **kwargs)

    np.testing.assert_array_equal(actual_long[: len(prefix_long)], expected_long)
    np.testing.assert_array_equal(actual_short[: len(prefix_short)], expected_short)
    np.testing.assert_allclose(
        actual_diagnostics["gap_ratio"][: len(prefix_long)],
        expected_diagnostics["gap_ratio"],
        equal_nan=True,
    )
