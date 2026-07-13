from __future__ import annotations

import numpy as np

from training.search_closed_excursion_holonomy_alpha import (
    build_signals,
    closed_excursion_features,
    fit_threshold,
)


def _state(
    close: list[float],
    return_z: list[float],
    flow: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = len(close)
    return (
        np.asarray(close, dtype=float),
        np.asarray(return_z, dtype=float),
        np.asarray(flow, dtype=float),
        np.full(size, 0.01),
    )


def test_upward_closed_loop_with_buy_inventory_resolves_short() -> None:
    state = _state(
        [100.0, 103.0, 104.0, 102.0, 100.0],
        [0.0, 3.0, 0.2, -0.2, -0.5],
        [0.0, 2.0, 2.0, 1.0, 1.0],
    )
    features = closed_excursion_features(
        *state,
        departure_z=2.0,
        max_age=10,
        min_age=4,
    )

    assert features.loc[4, "excursion_side"] == 1
    assert features.loc[4, "all_loop_side"] == 1
    assert features.loc[4, "cumulative_flow"] > 0.0
    long_active, short_active = build_signals(features, "cumulative_flow", 0.0)
    assert not long_active.any()
    np.testing.assert_array_equal(np.flatnonzero(short_active), [4])


def test_downward_closed_loop_with_sell_inventory_resolves_long() -> None:
    state = _state(
        [100.0, 97.0, 96.0, 98.0, 100.0],
        [0.0, -3.0, -0.2, 0.2, 0.5],
        [0.0, -2.0, -2.0, -1.0, -1.0],
    )
    features = closed_excursion_features(
        *state,
        departure_z=2.0,
        max_age=10,
        min_age=4,
    )

    long_active, short_active = build_signals(features, "cumulative_flow", 0.0)
    np.testing.assert_array_equal(np.flatnonzero(long_active), [4])
    assert not short_active.any()


def test_opposite_net_flow_keeps_price_loop_control_but_no_holonomy_score() -> None:
    state = _state(
        [100.0, 103.0, 104.0, 102.0, 100.0],
        [0.0, 3.0, 0.2, -0.2, -0.5],
        [0.0, -2.0, -2.0, -1.0, -1.0],
    )
    features = closed_excursion_features(
        *state,
        departure_z=2.0,
        max_age=10,
        min_age=4,
    )

    assert features.loc[4, "all_loop_side"] == 1
    assert features.loc[4, "excursion_side"] == 0
    assert np.isnan(features.loc[4, "cumulative_flow"])


def test_invalid_bar_resets_excursion() -> None:
    state = list(
        _state(
            [100.0, 103.0, np.nan, 102.0, 100.0],
            [0.0, 3.0, np.nan, -0.2, -0.5],
            [0.0, 2.0, np.nan, 1.0, 1.0],
        )
    )
    features = closed_excursion_features(
        *state,
        departure_z=2.0,
        max_age=10,
        min_age=4,
    )

    assert not (features["all_loop_side"] != 0).any()


def test_prefix_is_suffix_independent_and_flip_is_exact() -> None:
    prefix = [100.0, 103.0, 104.0, 102.0, 100.0]
    return_prefix = [0.0, 3.0, 0.2, -0.2, -0.5]
    flow_prefix = [0.0, 2.0, 2.0, 1.0, 1.0]
    first = _state(prefix + [200.0, 50.0], return_prefix + [10.0, -10.0], flow_prefix + [10.0, -10.0])
    second = _state(prefix + [50.0, 200.0], return_prefix + [-10.0, 10.0], flow_prefix + [-10.0, 10.0])

    first_features = closed_excursion_features(
        *first, departure_z=2.0, max_age=10, min_age=4
    )
    second_features = closed_excursion_features(
        *second, departure_z=2.0, max_age=10, min_age=4
    )
    for column in first_features:
        np.testing.assert_allclose(
            first_features[column].to_numpy()[: len(prefix)],
            second_features[column].to_numpy()[: len(prefix)],
            equal_nan=True,
        )
    long_active, short_active = build_signals(first_features, "cumulative_flow", 0.0)
    flip_long, flip_short = build_signals(
        first_features, "cumulative_flow", 0.0, flip=True
    )
    np.testing.assert_array_equal(flip_long, short_active)
    np.testing.assert_array_equal(flip_short, long_active)


def test_fit_threshold_ignores_nonfit_suffix() -> None:
    values = np.r_[np.arange(1.0, 101.0), [1_000_000.0, 2_000_000.0]]
    fit_mask = np.r_[np.ones(100, dtype=bool), np.zeros(2, dtype=bool)]

    threshold, count = fit_threshold(values, fit_mask, 0.90)

    np.testing.assert_allclose(threshold, np.quantile(np.arange(1.0, 101.0), 0.90))
    assert count == 100
