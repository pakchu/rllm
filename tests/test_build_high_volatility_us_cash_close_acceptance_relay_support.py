import numpy as np
import pandas as pd

from training import build_high_volatility_us_cash_close_acceptance_relay_support as support


def test_strict_prior_midrank_excludes_current() -> None:
    ranked = support.strict_prior_midrank(pd.Series(np.arange(127, dtype=float)))
    assert ranked.iloc[:126].isna().all() and ranked.iloc[126] == 1.0


def test_session_bounds_are_dst_aware() -> None:
    winter = support.session_bounds(pd.Timestamp("2024-01-02")); summer = support.session_bounds(pd.Timestamp("2024-07-01"))
    assert winter == (pd.Timestamp("2024-01-02T14:30:00Z"), pd.Timestamp("2024-01-02T21:00:00Z"))
    assert summer == (pd.Timestamp("2024-07-01T13:30:00Z"), pd.Timestamp("2024-07-01T20:00:00Z"))


def test_clock_follows_accepted_session_direction() -> None:
    states = pd.DataFrame({"session_date": ["2024-01-02", "2024-01-03", "2024-01-04"], "session_start": pd.to_datetime(["2024-01-02T14:30:00Z", "2024-01-03T14:30:00Z", "2024-01-04T14:30:00Z"]), "decision_time": pd.to_datetime(["2024-01-02T21:00:00Z", "2024-01-03T21:00:00Z", "2024-01-04T21:00:00Z"]), "source_valid": True, "session_return": [0.1, 0.0, -0.1], "session_efficiency": 0.9, "terminal_location": [0.9, 0.5, 0.1], "absolute_return_rank": [0.9, 0.0, 0.9], "efficiency_rank": 0.9, "btc_variation": 1.0, "btc_variation_rank": 0.9})
    clock = support.build_clock(states)
    assert list(clock.side) == [1, -1]
    assert (clock.entry_time - clock.decision_time).eq(pd.Timedelta(minutes=5)).all()
