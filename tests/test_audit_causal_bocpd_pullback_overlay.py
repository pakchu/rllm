from __future__ import annotations

import numpy as np
import pandas as pd

from training.audit_causal_bocpd_pullback_overlay import (
    build_bocpd_state,
    completed_hour_features,
    exact_hour_map,
)


def test_completed_hour_features_excludes_unfinished_boundary_bar() -> None:
    dates = pd.date_range("2023-01-01", periods=25, freq="5min")
    close = np.r_[np.full(12, 100.0), np.full(12, 110.0), 1_000_000.0]
    market = pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "quote_asset_volume": np.full(len(dates), 100.0),
            "taker_buy_quote": np.full(len(dates), 50.0),
        }
    )

    features = completed_hour_features(market)

    assert list(features.index) == [pd.Timestamp("2023-01-01 01:00:00"), pd.Timestamp("2023-01-01 02:00:00")]
    assert np.isclose(features.loc["2023-01-01 02:00:00", "ret1"], np.log(1.1))


def test_exact_hour_map_does_not_carry_state_into_later_five_minute_rows() -> None:
    dates = pd.Series(pd.to_datetime(["2023-01-01 01:00", "2023-01-01 01:05"]))
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01 01:00"]),
            "primary": [0.25],
            "short_mass": [0.75],
            "secondary": [-0.5],
        }
    )

    mapped = exact_hour_map(dates, output)

    assert mapped.loc[0, "primary"] == 0.25
    assert np.isnan(mapped.loc[1, "primary"])


def test_build_bocpd_state_marks_missing_rows_unavailable() -> None:
    mapped = pd.DataFrame(
        {
            "primary": [-2.0, 0.0, 2.0, np.nan],
            "short_mass": [0.2, 0.8, 0.2, 0.8],
            "secondary": [0.2, 0.2, 0.8, 0.8],
        }
    )
    thresholds = {
        "primary_low": -1.0,
        "primary_high": 1.0,
        "short_mass_high": 0.5,
        "secondary_high": 0.5,
    }

    states = build_bocpd_state(mapped, thresholds)

    assert states.tolist() == [0, 6, 9, -1]
