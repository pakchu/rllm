from __future__ import annotations

import numpy as np

from training.search_positioning_inventory_transfer_alpha import (
    inventory_transfer_signals,
)
from training.search_positioning_lifecycle_hazard_alpha import lifecycle_signals


def test_conserved_inventory_follows_resolution_direction() -> None:
    values = np.array([1.6, 1.7, 1.4, 0.2, -0.1])
    oi = np.array([100.0, 101.0, 102.0, 103.0, 101.0])

    long_active, short_active, diagnostics = inventory_transfer_signals(
        values,
        np.ones(len(values), dtype=bool),
        oi,
        min_age=5,
    )

    assert not long_active.any()
    np.testing.assert_array_equal(np.flatnonzero(short_active), [4])
    assert diagnostics["episode_start_index"][4] == 0
    assert diagnostics["inventory_conserved"][4]


def test_inventory_contraction_fades_resolution_direction() -> None:
    values = np.array([1.6, 1.7, 1.4, 0.2, -0.1])
    oi = np.array([100.0, 99.0, 98.0, 97.0, 96.0])

    long_active, short_active, diagnostics = inventory_transfer_signals(
        values,
        np.ones(len(values), dtype=bool),
        oi,
        min_age=5,
    )

    np.testing.assert_array_equal(np.flatnonzero(long_active), [4])
    assert not short_active.any()
    assert diagnostics["oi_log_change"][4] < 0.0
    assert not diagnostics["inventory_conserved"][4]


def test_invalid_row_resets_inventory_episode() -> None:
    values = np.array([1.6, 1.7, 1.4, 0.2, -0.1])
    valid = np.array([True, True, False, True, True])
    oi = np.full(len(values), 100.0)

    long_active, short_active, _ = inventory_transfer_signals(
        values,
        valid,
        oi,
        min_age=5,
    )

    assert not long_active.any() and not short_active.any()


def test_prefix_is_suffix_independent_and_controls_are_exact() -> None:
    prefix = np.array([1.6, 1.7, 1.4, 0.2, -0.1])
    first = np.r_[prefix, [100.0, -100.0]]
    second = np.r_[prefix, [-100.0, 100.0]]
    oi = np.array([100.0, 101.0, 102.0, 103.0, 101.0, 99.0, 98.0])
    valid = np.ones(len(first), dtype=bool)

    first_long, first_short, first_diagnostics = inventory_transfer_signals(
        first, valid, oi, min_age=5
    )
    second_long, second_short, second_diagnostics = inventory_transfer_signals(
        second, valid, oi, min_age=5
    )
    flip_long, flip_short, _ = inventory_transfer_signals(
        first, valid, oi, min_age=5, flip=True
    )
    invert_long, invert_short, _ = inventory_transfer_signals(
        first, valid, oi, min_age=5, invert_oi=True
    )

    np.testing.assert_array_equal(first_long[: len(prefix)], second_long[: len(prefix)])
    np.testing.assert_array_equal(first_short[: len(prefix)], second_short[: len(prefix)])
    for key in first_diagnostics:
        np.testing.assert_allclose(
            first_diagnostics[key][: len(prefix)],
            second_diagnostics[key][: len(prefix)],
            equal_nan=True,
        )
    np.testing.assert_array_equal(flip_long, first_short)
    np.testing.assert_array_equal(flip_short, first_long)
    np.testing.assert_array_equal(invert_long, first_short)
    np.testing.assert_array_equal(invert_short, first_long)


def test_ignore_oi_matches_parent_lifecycle_zero_cross() -> None:
    values = np.array([1.6, 1.7, 1.4, 0.2, -0.1])
    valid = np.ones(len(values), dtype=bool)
    oi = np.array([100.0, 99.0, 98.0, 97.0, 96.0])

    actual_long, actual_short, _ = inventory_transfer_signals(
        values,
        valid,
        oi,
        min_age=5,
        ignore_oi=True,
    )
    expected_long, expected_short, _ = lifecycle_signals(
        values,
        valid,
        min_age=5,
        trigger="zero_cross",
    )

    np.testing.assert_array_equal(actual_long, expected_long)
    np.testing.assert_array_equal(actual_short, expected_short)
