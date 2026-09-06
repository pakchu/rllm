import numpy as np
import pandas as pd
from training import build_high_volatility_regional_bank_close_location_relay_support as support


def test_strict_prior_midrank_excludes_current():
    values = pd.Series(np.arange(181, dtype=float))
    ranked = support.strict_prior_midrank(values)
    assert ranked.iloc[:180].isna().all()
    assert ranked.iloc[180] == 1.0


def states(location=0.75, cash_return=0.01, rank=0.8):
    return pd.DataFrame({"session_date": pd.to_datetime(["2023-07-03"]), "decision_time": pd.to_datetime(["2023-07-03T23:00:00Z"]), "source_valid": [True], "kre_cash_return": [cash_return], "close_location": [location], "btc_variation": [0.2], "btc_variation_rank": [rank]})


def test_primary_side_threshold_and_timing():
    clock = support.build_clock(states())
    assert clock.side.tolist() == [1]
    assert clock.entry_time.iloc[0] == pd.Timestamp("2023-07-03T23:05:00Z")
    assert clock.exit_time.iloc[0] == pd.Timestamp("2023-07-04T23:05:00Z")
    assert support.build_clock(states(location=-0.8)).side.tolist() == [-1]
    assert support.build_clock(states(location=0.4)).empty


def test_controls_are_diagnostic_only():
    assert support.build_clock(states(location=0.4), "any_nonzero_close_location").side.tolist() == [1]
    assert support.build_clock(states(location=0.8, cash_return=-0.01), "cash_return_direction").side.tolist() == [-1]
    assert support.build_clock(states(rank=0.2)).empty
    assert support.build_clock(states(rank=0.2), "no_variation_gate").side.tolist() == [1]
    assert support.build_clock(states(), "direction_flip").side.tolist() == [-1]
