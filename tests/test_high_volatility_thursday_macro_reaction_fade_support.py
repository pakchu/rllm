import numpy as np
import pandas as pd

from training import build_high_volatility_thursday_macro_reaction_fade_support as support


def test_thursday_anchors_follow_new_york_dst():
    anchors = support.thursday_anchors(pd.Timestamp("2024-03-01T00:00:00Z"), pd.Timestamp("2024-03-22T00:00:00Z"))
    assert anchors.tolist() == [pd.Timestamp("2024-03-07T13:30:00Z"), pd.Timestamp("2024-03-14T12:30:00Z"), pd.Timestamp("2024-03-21T12:30:00Z")]


def test_rank_is_strict_prior_and_current_excluded():
    dates = pd.date_range("2021-01-01", "2023-01-15", freq="5min", tz="UTC", inclusive="left")
    price = np.exp(np.arange(len(dates)) * 1e-7) * 100
    market = pd.DataFrame({"date": dates, "open": price, "high": price * 1.001, "low": price * .999, "close": price})
    states = support.score_states(market)
    assert states.variation_rank.iloc[:60].isna().all()
    assert np.isfinite(states.variation_rank.iloc[60])


def test_primary_fades_and_fixed_controls_keep_frozen_laws():
    states = pd.DataFrame([{"anchor_time": pd.Timestamp("2023-07-06T12:30:00Z"), "decision_time": pd.Timestamp("2023-07-06T13:30:00Z"), "reaction_return": .01, "half_hour_return": -.01, "pre_anchor_variation": .1, "variation_rank": .8}])
    assert support.build_clock(states).side.tolist() == [-1]
    assert support.build_clock(states, "reaction_continuation").side.tolist() == [1]
    assert support.build_clock(states, "half_hour_reaction_fade").side.tolist() == [1]
    assert support.build_clock(states, "same_clock_forced_long").side.tolist() == [1]
