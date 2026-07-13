from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_cme_offshore_debt_handoff_alpha import (
    fit_threshold,
    prepare_cftc_reports,
    prior_z,
    signals,
)


def test_cftc_report_is_delayed_eight_days_and_cut_off() -> None:
    raw = pd.DataFrame(
        {
            "report_date_as_yyyy_mm_dd": ["2023-12-19", "2023-12-26"],
            "open_interest_all": [1, 1],
        }
    )
    result = prepare_cftc_reports(raw)
    assert result["release_time"].tolist() == [pd.Timestamp("2023-12-27")]


def test_prior_z_prefix_does_not_depend_on_future_suffix() -> None:
    prefix = pd.Series(np.sin(np.arange(300, dtype=float) / 19.0))
    full = pd.concat([prefix, pd.Series([1_000_000.0] * 20)], ignore_index=True)
    pd.testing.assert_series_equal(
        prior_z(full, 52, 26).iloc[: len(prefix)].reset_index(drop=True),
        prior_z(prefix, 52, 26).reset_index(drop=True),
    )


def test_fit_threshold_ignores_2023_selection() -> None:
    fit_dates = pd.date_range("2020-10-15", periods=80, freq="7D")
    selection = pd.date_range("2023-01-01", periods=20, freq="7D")
    events = pd.DataFrame(
        {
            "release_time": fit_dates.append(selection),
            "handoff": np.r_[np.arange(80, dtype=float), np.full(20, 1_000_000.0)],
        }
    )
    assert fit_threshold(events, 0.8) == float(pd.Series(np.arange(80, dtype=float)).quantile(0.8))


def test_signal_fades_offshore_receiver_and_flip_is_exact() -> None:
    events = pd.DataFrame(
        {
            "handoff": [2.0, 3.0, 0.0],
            "offshore_z": [2.0, -2.0, 1.0],
            "offshore": [0.2, -0.3, 0.1],
            "signal_pos": [2, 5, 7],
        }
    )
    long_signal, short_signal = signals(events, 10, 1.0)
    assert np.flatnonzero(long_signal).tolist() == [5]
    assert np.flatnonzero(short_signal).tolist() == [2]
    flip_long, flip_short = signals(events, 10, 1.0, flip=True)
    np.testing.assert_array_equal(flip_long, short_signal)
    np.testing.assert_array_equal(flip_short, long_signal)
