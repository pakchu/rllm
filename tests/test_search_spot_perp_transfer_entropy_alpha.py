from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_spot_perp_transfer_entropy_alpha import (
    discretize_returns,
    fit_quantile,
    rolling_transfer_entropy,
    signals,
)


def test_discretize_returns_prefix_does_not_depend_on_future_suffix() -> None:
    prefix = pd.Series(np.sin(np.arange(420, dtype=float) / 17.0) / 100.0)
    full = pd.concat([prefix, pd.Series([100.0] * 30)], ignore_index=True)

    np.testing.assert_array_equal(
        discretize_returns(full, 1.0)[: len(prefix)],
        discretize_returns(prefix, 1.0),
    )


def test_transfer_entropy_excludes_current_and_future_states() -> None:
    x = np.resize(np.array([0, 1, 2, 1], dtype=np.int8), 40)
    y = np.resize(np.array([1, 2, 0, 2], dtype=np.int8), 40)
    decision = np.zeros(40, dtype=bool)
    decision[20] = True

    baseline = rolling_transfer_entropy(x, y, window=16, decision=decision)
    changed_x = x.copy()
    changed_y = y.copy()
    changed_x[20:] = 2
    changed_y[20:] = 0
    changed = rolling_transfer_entropy(changed_x, changed_y, window=16, decision=decision)

    assert np.isfinite(baseline[20])
    assert changed[20] == baseline[20]


def test_fit_quantile_ignores_2023_selection_values() -> None:
    fit_dates = pd.date_range("2020-06-01", periods=6_000, freq="3h")
    selection_dates = pd.date_range("2023-01-01", periods=200, freq="3h")
    dates = pd.Series(fit_dates.append(selection_dates))
    fit_values = np.arange(6_000, dtype=float)
    features = pd.DataFrame(
        {
            "decision": True,
            "te_advantage": np.r_[fit_values, np.full(200, 1_000_000.0)],
        }
    )

    assert fit_quantile(features, dates, "te_advantage", 0.7) == float(
        pd.Series(fit_values).quantile(0.7)
    )


def test_signals_follow_gap_and_flip_is_exact() -> None:
    features = pd.DataFrame(
        {
            "te_advantage": [2.0, 2.0, 0.1, 2.0],
            "lead_gap_z": [3.0, -3.0, 3.0, 0.1],
            "spot_move": [1.0, -1.0, 1.0, 1.0],
            "decision": [True, True, True, True],
        }
    )

    long_signal, short_signal = signals(features, 1.0, 1.0, "gap_only")
    np.testing.assert_array_equal(long_signal, [True, False, False, False])
    np.testing.assert_array_equal(short_signal, [False, True, False, False])
    flip_long, flip_short = signals(features, 1.0, 1.0, "gap_only", flip=True)
    np.testing.assert_array_equal(flip_long, short_signal)
    np.testing.assert_array_equal(flip_short, long_signal)
