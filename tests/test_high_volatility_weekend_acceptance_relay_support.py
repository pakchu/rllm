import numpy as np
import pandas as pd

from training import build_high_volatility_weekend_acceptance_relay_support as support


def test_geometry_and_directional_acceptance():
    dates = pd.date_range("2024-01-06", periods=576, freq="5min", tz="UTC")
    close = np.linspace(100., 120., len(dates))
    frame = pd.DataFrame({"date": dates, "open": close, "high": close + 1., "low": close - 1., "close": close})
    value = support.geometry(frame)
    assert value is not None
    displacement, range_log, location = value
    assert displacement > 0 and range_log > 0 and location >= .75


def test_rank_is_strict_prior_and_current_is_excluded():
    dates = pd.date_range("2022-01-01", "2023-04-04", freq="5min", tz="UTC", inclusive="left")
    step = np.arange(len(dates), dtype=float)
    price = np.exp(step * 1e-7) * 100
    market = pd.DataFrame({"date": dates, "open": price, "high": price * 1.001, "low": price * .999, "close": price})
    states = support.score_states(market)
    assert states.range_rank.iloc[:60].isna().all()
    assert np.isfinite(states.range_rank.iloc[60])


def test_primary_and_controls_use_frozen_side_laws():
    row = {"decision_time": pd.Timestamp("2023-07-03T00:00:00Z"), "displacement": .1, "range_log": .2, "close_location": .9, "range_rank": .8}
    full = pd.DataFrame([row]); sunday = full.copy()
    primary = support.build_clock(full, sunday)
    assert primary.side.tolist() == [1]
    assert support.build_clock(full, sunday, "direction_flip").side.tolist() == [-1]
    assert support.build_clock(full, sunday, "same_clock_forced_long").side.tolist() == [1]
