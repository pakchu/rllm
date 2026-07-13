from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_cftc_release_assimilation_alpha import (
    assimilation_signals,
    build_assimilation_events,
    fit_surprise_threshold,
    participant_surprise,
)


def _reports(rows: int = 130) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    report_date = pd.Series(pd.date_range("2019-01-01", periods=rows, freq="7D"))
    open_interest = 10_000.0 + 10.0 * index
    leveraged_net = 0.1 * np.sin(index / 5.0) + 0.001 * index
    asset_net = 0.08 * np.cos(index / 7.0) - 0.0005 * index
    return pd.DataFrame(
        {
            "report_date": report_date,
            "release_time": report_date + pd.Timedelta(days=8),
            "open_interest_all": open_interest,
            "lev_money_positions_long": open_interest * (0.4 + leveraged_net / 2.0),
            "lev_money_positions_short": open_interest * (0.4 - leveraged_net / 2.0),
            "asset_mgr_positions_long": open_interest * (0.3 + asset_net / 2.0),
            "asset_mgr_positions_short": open_interest * (0.3 - asset_net / 2.0),
        }
    )


def test_participant_surprise_prefix_is_suffix_independent() -> None:
    reports = _reports()
    expected = participant_surprise(reports, "leveraged_money")
    changed = reports.copy()
    changed.loc[110:, "lev_money_positions_long"] *= 10.0
    actual = participant_surprise(changed, "leveraged_money")

    np.testing.assert_allclose(actual.iloc[:110], expected.iloc[:110], equal_nan=True)


def test_event_return_uses_report_to_conservative_release_path() -> None:
    reports = _reports()
    dates = pd.Series(pd.date_range("2018-12-01", "2021-12-31", freq="1D"))
    close = 100.0 * np.exp(0.001 * np.arange(len(dates)))
    market = pd.DataFrame({"close": close})

    events = build_assimilation_events(
        market,
        dates,
        reports,
        participant="leveraged_money",
    )

    row = events.iloc[-1]
    report_position = dates.searchsorted(row.report_date)
    release_position = dates.searchsorted(row.release_time)
    np.testing.assert_allclose(
        row.report_to_release_return,
        np.log(close[release_position] / close[report_position]),
    )
    assert row.signal_position == release_position


def test_assimilation_state_routes_are_symmetric_and_flip_is_exact() -> None:
    events = pd.DataFrame(
        {
            "position_surprise_z": [2.0, -2.0, 2.0, -2.0],
            "assimilation_fraction": [-0.2, -0.1, 1.2, 1.5],
            "signal_position": [1, 2, 3, 4],
        }
    )

    unpriced_long, unpriced_short = assimilation_signals(
        events, 8, threshold=1.0, state="unpriced"
    )
    over_long, over_short = assimilation_signals(
        events, 8, threshold=1.0, state="over_assimilated"
    )
    flip_long, flip_short = assimilation_signals(
        events, 8, threshold=1.0, state="unpriced", flip=True
    )

    np.testing.assert_array_equal(np.flatnonzero(unpriced_long), [1])
    np.testing.assert_array_equal(np.flatnonzero(unpriced_short), [2])
    np.testing.assert_array_equal(np.flatnonzero(over_long), [4])
    np.testing.assert_array_equal(np.flatnonzero(over_short), [3])
    np.testing.assert_array_equal(flip_long, unpriced_short)
    np.testing.assert_array_equal(flip_short, unpriced_long)


def test_fit_threshold_ignores_2023_values() -> None:
    fit_values = np.linspace(-2.0, 2.0, 120)
    events = pd.DataFrame(
        {
            "release_time": list(pd.date_range("2020-06-01", periods=120, freq="7D"))
            + [pd.Timestamp("2023-03-01"), pd.Timestamp("2023-09-01")],
            "position_surprise_z": list(fit_values) + [100.0, 200.0],
        }
    )

    actual = fit_surprise_threshold(events, 0.8)

    expected = np.quantile(np.abs(fit_values), 0.8)
    np.testing.assert_allclose(actual, expected)


def test_stale_release_control_moves_signal_forward_only() -> None:
    events = pd.DataFrame(
        {
            "position_surprise_z": [2.0],
            "assimilation_fraction": [-0.5],
            "signal_position": [10],
        }
    )

    long_active, short_active = assimilation_signals(
        events,
        10_000,
        threshold=1.0,
        state="unpriced",
        release_extra_weeks=4,
    )

    np.testing.assert_array_equal(np.flatnonzero(long_active), [10 + 4 * 7 * 288])
    assert not short_active.any()
