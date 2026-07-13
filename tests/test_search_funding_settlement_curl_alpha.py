from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_funding_settlement_curl_alpha import (
    WINDOWS,
    build_event_features,
    fit_quantile,
    prior_event_z,
)


def test_prior_event_z_prefix_is_independent_of_future_suffix() -> None:
    prefix = pd.Series(np.sin(np.arange(300, dtype=float) / 13.0))
    full = pd.concat([prefix, pd.Series([1_000_000.0] * 20)], ignore_index=True)
    pd.testing.assert_series_equal(
        prior_event_z(full).iloc[: len(prefix)].reset_index(drop=True),
        prior_event_z(prefix).reset_index(drop=True),
    )


def test_event_features_wait_for_exact_settlement_and_post_window() -> None:
    dates = pd.Series(pd.date_range("2022-01-01", periods=320, freq="5min"))
    premium = np.repeat(np.arange(27, dtype=float), 12)[: len(dates)]
    oi = np.exp(np.linspace(5.0, 5.5, len(dates)))
    market = pd.DataFrame({"premium_index": premium, "sum_open_interest": oi})
    events = pd.DataFrame(
        {
            "event_time": pd.date_range("2022-01-01 08:00:00.001", periods=1, freq="8h"),
            "funding_rate": [0.001],
        }
    )
    result = build_event_features(market, dates, events, pre_bars=12, post_bars=12)
    assert len(result) == 1
    # A settlement one millisecond after 08:00 cannot be assigned to 08:00.
    assert result.iloc[0]["known_bar_time"] == pd.Timestamp("2022-01-01 08:05")
    assert result.iloc[0]["signal_time"] == pd.Timestamp("2022-01-01 09:05")
    assert int(result.iloc[0]["signal_pos"]) == 109


def test_fit_quantile_ignores_2023_selection_and_future_rows() -> None:
    fit_start, fit_end = WINDOWS["fit"]
    fit_dates = pd.date_range(fit_start, periods=200, freq="D")
    selection_dates = pd.date_range(fit_end, periods=30, freq="D")
    events = pd.DataFrame(
        {
            "signal_time": fit_dates.append(selection_dates),
            "trap": np.r_[np.arange(200, dtype=float), np.full(30, 1_000_000.0)],
        }
    )
    assert fit_quantile(events, "trap", 0.5) == 99.5
