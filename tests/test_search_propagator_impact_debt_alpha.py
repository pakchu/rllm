from __future__ import annotations

import numpy as np

from training.search_propagator_impact_debt_alpha import (
    build_signals,
    fit_ridge,
    impact_debt,
    lag_matrix,
    prior_z,
)


def test_lag_matrix_uses_current_and_past_only() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])

    actual = lag_matrix(values, 2)

    np.testing.assert_allclose(
        actual,
        [[1.0, np.nan, np.nan], [2.0, 1.0, np.nan], [3.0, 2.0, 1.0], [4.0, 3.0, 2.0]],
        equal_nan=True,
    )


def test_impact_debt_is_remaining_future_kernel_mass() -> None:
    innovations = np.array([1.0, 0.0, 0.0, 0.0])
    kernel = np.array([0.5, 0.3, 0.2])

    debt, tail = impact_debt(innovations, kernel)

    np.testing.assert_allclose(tail, [0.5, 0.2])
    assert np.isnan(debt[0])
    np.testing.assert_allclose(debt[1:], [0.2, 0.0, 0.0])


def test_prior_z_prefix_is_suffix_independent() -> None:
    prefix = np.arange(1.0, 9.0)
    first = np.r_[prefix, [100.0, -100.0]]
    second = np.r_[prefix, [-100.0, 100.0]]

    first_z = prior_z(first, window=4)
    second_z = prior_z(second, window=4)

    np.testing.assert_allclose(first_z[: len(prefix)], second_z[: len(prefix)], equal_nan=True)


def test_ridge_fit_ignores_rows_outside_explicit_fit_mask() -> None:
    values = np.linspace(-1.0, 1.0, 20_100)
    predictors = np.column_stack([values, values**2])
    target = 0.2 + 0.7 * values - 0.3 * values**2
    fit_mask = np.zeros(len(values), dtype=bool)
    fit_mask[:20_000] = True
    changed_target = target.copy()
    changed_target[~fit_mask] = 1_000_000.0

    first, first_count = fit_ridge(predictors, target, fit_mask)
    second, second_count = fit_ridge(predictors, changed_target, fit_mask)

    np.testing.assert_allclose(first, second)
    assert first_count == second_count == 20_000


def test_onset_and_direction_flip_are_exact() -> None:
    values = np.array([0.0, 2.0, 2.2, 0.0, -2.0, -2.2])

    long_active, short_active = build_signals(values, 1.5, "onset")
    flip_long, flip_short = build_signals(values, 1.5, "onset", flip=True)

    np.testing.assert_array_equal(np.flatnonzero(long_active), [1])
    np.testing.assert_array_equal(np.flatnonzero(short_active), [4])
    np.testing.assert_array_equal(flip_long, short_active)
    np.testing.assert_array_equal(flip_short, long_active)


def test_state_mode_keeps_completed_extreme_state() -> None:
    values = np.array([0.0, 2.0, 2.2, 0.0, -2.0, -2.2])

    long_active, short_active = build_signals(values, 1.5, "state")

    np.testing.assert_array_equal(np.flatnonzero(long_active), [1, 2])
    np.testing.assert_array_equal(np.flatnonzero(short_active), [4, 5])
