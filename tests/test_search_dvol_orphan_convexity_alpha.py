from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_dvol_orphan_convexity_alpha import (
    _hourly_dvol_change,
    build_orphan_features,
    causal_rolling_residual,
    orphan_signals,
)


def test_rolling_residual_uses_only_prior_observations() -> None:
    x = np.arange(10, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    target = 2.0 + 3.0 * x
    target[-1] += 10.0

    residual = causal_rolling_residual(
        target,
        design,
        window=8,
        min_observations=4,
    )

    np.testing.assert_allclose(residual[4:9], 0.0, atol=1e-10)
    np.testing.assert_allclose(residual[9], 10.0, atol=1e-10)


def test_rolling_residual_prefix_is_suffix_independent() -> None:
    x = np.arange(20, dtype=float)
    design = np.column_stack([np.ones(len(x)), x, np.sin(x)])
    target = 1.0 + 0.2 * x + 0.5 * np.sin(x)
    expected = causal_rolling_residual(target, design, window=8, min_observations=4)

    changed_target = target.copy()
    changed_design = design.copy()
    changed_target[15:] *= -100.0
    changed_design[15:, 1:] *= 50.0
    actual = causal_rolling_residual(changed_target, changed_design, window=8, min_observations=4)

    np.testing.assert_allclose(actual[:15], expected[:15], equal_nan=True)


def test_hourly_dvol_change_rejects_nonconsecutive_gap() -> None:
    market = pd.DataFrame(
        {
            "close_time": pd.to_datetime(
                ["2023-01-01 01:00", "2023-01-01 02:00", "2023-01-01 04:00"]
            ),
            "dvol_close": [100.0, 110.0, 121.0],
        }
    )
    update = np.ones(len(market), dtype=bool)

    actual = _hourly_dvol_change(market, update)

    assert np.isnan(actual[0])
    np.testing.assert_allclose(actual[1], np.log(1.1))
    assert np.isnan(actual[2])


def test_signal_threshold_uses_fit_reference_only_and_flip_is_exact() -> None:
    features = pd.DataFrame(
        {
            "orphan_residual_30d": [0.0, 1.0, 2.0, 3.0, 100.0, 101.0],
            "taker_flow_1h": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            "fit_reference": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "dvol_update": [1.0] * 6,
        }
    )

    long_active, short_active, threshold = orphan_signals(
        features,
        residual_days=30,
        tail=0.75,
        direction_proxy="taker_flow_1h",
    )
    flip_long, flip_short, _ = orphan_signals(
        features,
        residual_days=30,
        tail=0.75,
        direction_proxy="taker_flow_1h",
        flip=True,
    )

    assert threshold == np.quantile([0.0, 1.0, 2.0, 3.0], 0.75)
    np.testing.assert_array_equal(np.flatnonzero(long_active), [4])
    np.testing.assert_array_equal(np.flatnonzero(short_active), [3, 5])
    np.testing.assert_array_equal(flip_long, short_active)
    np.testing.assert_array_equal(flip_short, long_active)


def test_incomplete_spot_bundle_cannot_enter_residual_model() -> None:
    rows = 5_000
    index = np.arange(rows, dtype=float)
    dates = pd.Series(pd.date_range("2021-04-15", periods=rows, freq="5min"))
    close = 100.0 * np.exp(np.cumsum(0.0001 * np.sin(index / 17.0)))
    hour = (index // 12).astype(int)
    market = pd.DataFrame(
        {
            "close": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "premium_index_1m_close": 0.0001 * np.sin(index / 29.0),
            "quote_asset_volume": np.full(rows, 1_000_000.0),
            "taker_buy_quote": 500_000.0 + 10_000.0 * np.sin(index / 11.0),
            "spot_close": close * 0.9999,
            "premium_available": np.ones(rows, dtype=bool),
            "spot_available": np.ones(rows, dtype=bool),
            "close_time": dates.dt.floor("h"),
            "dvol_close": 80.0 + 0.01 * hour + 0.1 * np.sin(hour / 13.0),
        }
    )
    incomplete_update = 4_500
    market.loc[incomplete_update, "spot_available"] = False

    features = build_orphan_features(market, dates)

    assert np.isfinite(features.loc[incomplete_update - 12, "orphan_residual_30d"])
    assert np.isnan(features.loc[incomplete_update, "orphan_residual_30d"])
