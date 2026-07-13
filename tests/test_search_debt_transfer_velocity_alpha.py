from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_debt_transfer_velocity_alpha import (
    WINDOWS,
    fit_threshold,
    prior_z,
    signals,
)


def test_prior_z_prefix_is_future_suffix_invariant() -> None:
    prefix = pd.Series(np.cos(np.arange(800, dtype=float) / 31.0))
    full = pd.concat([prefix, pd.Series([999_999.0] * 30)], ignore_index=True)
    pd.testing.assert_series_equal(
        prior_z(full, 288).iloc[: len(prefix)].reset_index(drop=True),
        prior_z(prefix, 288).reset_index(drop=True),
    )


def test_fit_threshold_uses_fit_window_only() -> None:
    fit_dates = pd.date_range(WINDOWS["fit"][0], periods=1_200, freq="12h")
    future_dates = pd.date_range(WINDOWS["fit"][1], periods=100, freq="12h")
    dates = pd.Series(fit_dates.append(future_dates))
    score = pd.Series(np.r_[np.arange(1, 1_201, dtype=float), np.full(100, 1_000_000.0)])
    expected = float(pd.Series(np.arange(1, 1_201, dtype=float)).quantile(0.9))
    assert fit_threshold(score, dates, 0.9) == expected


def test_signal_fades_receiver_and_flip_is_exact() -> None:
    features = pd.DataFrame(
        {
            "score": [0.0, 2.0, 2.0, 0.0, 3.0],
            "transfer_velocity": [1.0, 1.0, 1.0, -1.0, -1.0],
        }
    )
    long_signal, short_signal = signals(features, 1.0)
    np.testing.assert_array_equal(long_signal, [False, False, False, False, True])
    np.testing.assert_array_equal(short_signal, [False, True, False, False, False])
    flip_long, flip_short = signals(features, 1.0, flip=True)
    np.testing.assert_array_equal(flip_long, short_signal)
    np.testing.assert_array_equal(flip_short, long_signal)
