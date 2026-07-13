from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_funding_age_rollover_transfer_alpha import (
    build_features,
    fit_abs_threshold,
    prior_z,
    signals,
)


def _market(n: int) -> pd.DataFrame:
    x = np.arange(n, dtype=float)
    close = 100.0 * np.exp(0.0001 * np.sin(x / 31.0).cumsum())
    return pd.DataFrame(
        {
            "close": close,
            "sum_open_interest": 100_000.0 + 20.0 * np.sin(x / 19.0) + x * 0.01,
            "count_long_short_ratio": np.exp(0.05 * np.sin(x / 23.0)),
            "quote_asset_volume": np.full(n, 1_000_000.0),
            "taker_buy_quote": 500_000.0 + 50_000.0 * np.sin(x / 17.0),
        }
    )


def test_prior_z_prefix_is_future_suffix_invariant() -> None:
    prefix = pd.Series(np.sin(np.arange(2_500, dtype=float) / 29.0))
    full = pd.concat([prefix, pd.Series([1_000_000.0] * 100)], ignore_index=True)
    pd.testing.assert_series_equal(
        prior_z(full).iloc[: len(prefix)].reset_index(drop=True),
        prior_z(prefix).reset_index(drop=True),
    )


def test_cohort_state_prefix_is_future_suffix_invariant() -> None:
    n, prefix = 3_300, 3_100
    market = _market(n)
    rates = np.full(n, np.nan)
    rates[96::96] = 0.0001
    full = build_features(market, rates, min_age_settlements=1, half_life_bars=288)
    short = build_features(market.iloc[:prefix].copy(), rates[:prefix], min_age_settlements=1, half_life_bars=288)
    pd.testing.assert_frame_equal(full.iloc[:prefix].reset_index(drop=True), short.reset_index(drop=True))


def test_signal_uses_signed_transfer_and_flip_is_exact() -> None:
    features = pd.DataFrame({"score": [0.0, 2.0, 2.0, 0.0, -3.0]})
    long_signal, short_signal = signals(features, 1.0)
    np.testing.assert_array_equal(long_signal, [False, True, False, False, False])
    np.testing.assert_array_equal(short_signal, [False, False, False, False, True])
    flip_long, flip_short = signals(features, 1.0, flip=True)
    np.testing.assert_array_equal(flip_long, short_signal)
    np.testing.assert_array_equal(flip_short, long_signal)


def test_fit_threshold_ignores_selection_rows() -> None:
    fit_dates = pd.date_range("2020-10-15", periods=12_000, freq="5min")
    selection = pd.date_range("2023-01-01", periods=100, freq="5min")
    dates = pd.Series(fit_dates.append(selection))
    score = pd.Series(np.r_[np.arange(12_000, dtype=float), np.full(100, 1_000_000.0)])
    assert fit_abs_threshold(score, dates, 0.9) == float(pd.Series(np.arange(12_000, dtype=float)).quantile(0.9))
