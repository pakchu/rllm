from __future__ import annotations

import numpy as np

from training.build_high_volatility_value_area_rejection_relay_support import contiguous_value_area


def test_contiguous_value_area_expands_from_lower_tied_poc() -> None:
    prices = np.exp(np.linspace(0.02, 0.98, 24))
    weights = np.ones(24)
    lower, upper, poc_lower, poc_upper = contiguous_value_area(
        prices, weights, low=1.0, high=np.e, share=0.70, bins=24
    )

    assert np.isclose(poc_lower, 1.0)
    assert np.isclose(poc_upper, np.exp(1 / 24))
    assert np.isclose(lower, 1.0)
    assert np.isclose(upper, np.exp(17 / 24))


def test_contiguous_value_area_prefers_larger_adjacent_volume() -> None:
    prices = np.exp(np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
    weights = np.array([1.0, 2.0, 10.0, 8.0, 1.0])
    lower, upper, poc_lower, poc_upper = contiguous_value_area(
        prices, weights, low=1.0, high=np.e, share=0.70, bins=5
    )

    assert np.isclose(poc_lower, np.exp(0.4))
    assert np.isclose(poc_upper, np.exp(0.6))
    assert np.isclose(lower, np.exp(0.4))
    assert np.isclose(upper, np.exp(0.8))


def test_contiguous_value_area_rejects_invalid_inputs() -> None:
    result = contiguous_value_area(
        np.array([1.0, 2.0]), np.array([1.0, 0.0]), low=1.0, high=2.0
    )
    assert all(np.isnan(value) for value in result)
