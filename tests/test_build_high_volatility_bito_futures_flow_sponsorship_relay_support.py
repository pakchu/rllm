import numpy as np
import pandas as pd
from training import build_high_volatility_bito_futures_flow_sponsorship_relay_support as support


def test_strict_prior_midrank_excludes_current():
    values = pd.Series(np.arange(181, dtype=float))
    ranked = support.strict_prior_midrank(values)
    assert ranked.iloc[:180].isna().all()
    assert ranked.iloc[180] == 1.0


def states(cash_return=0.01, volume_rank=0.8, variation_rank=0.8):
    return pd.DataFrame({"session_date": pd.to_datetime(["2023-07-03"]), "decision_time": pd.to_datetime(["2023-07-03T23:00:00Z"]), "source_valid": [True], "bito_cash_return": [cash_return], "bito_volume": [1000.0], "bito_relative_volume_rank": [volume_rank], "btc_variation": [0.2], "btc_variation_rank": [variation_rank]})


def test_primary_side_and_timing():
    clock = support.build_clock(states())
    assert clock.side.tolist() == [1]
    assert clock.entry_time.iloc[0] == pd.Timestamp("2023-07-03T23:05:00Z")
    assert clock.exit_time.iloc[0] == pd.Timestamp("2023-07-04T23:05:00Z")
    assert support.build_clock(states(cash_return=-0.01)).side.tolist() == [-1]
    assert support.build_clock(states(volume_rank=0.4)).empty


def test_controls_are_diagnostic_only():
    assert support.build_clock(states(volume_rank=0.4), "no_relative_volume_gate").side.tolist() == [1]
    assert support.build_clock(states(variation_rank=0.2)).empty
    assert support.build_clock(states(variation_rank=0.2), "no_variation_gate").side.tolist() == [1]
    assert support.build_clock(states(), "direction_flip").side.tolist() == [-1]
