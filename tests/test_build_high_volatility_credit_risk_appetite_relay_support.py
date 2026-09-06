import numpy as np
import pandas as pd

from training import build_high_volatility_credit_risk_appetite_relay_support as support


def test_strict_prior_midrank_excludes_current():
    values = pd.Series(np.arange(181, dtype=float))
    ranked = support.strict_prior_midrank(values)
    assert ranked.iloc[:180].isna().all()
    assert ranked.iloc[180] == 1.0


def test_relative_credit_clock_side_and_timing():
    states = pd.DataFrame({
        "session_date": pd.to_datetime(["2023-07-03", "2023-07-05"]),
        "decision_time": pd.to_datetime(["2023-07-03T23:00:00Z", "2023-07-05T23:00:00Z"]),
        "source_valid": [True, True], "hyg_return": [0.01, -0.02], "lqd_return": [0.002, -0.005],
        "relative_credit_return": [0.008, -0.015], "btc_variation": [0.2, 0.3],
        "btc_variation_rank": [0.7, 0.8],
    })
    clock = support.build_clock(states)
    assert clock.side.tolist() == [1, -1]
    assert (clock.entry_time - clock.decision_time).eq(pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time - clock.entry_time).eq(pd.Timedelta(hours=24)).all()


def test_diagnostic_directions_do_not_change_primary_clock():
    states = pd.DataFrame({
        "session_date": pd.to_datetime(["2023-07-03"]),
        "decision_time": pd.to_datetime(["2023-07-03T23:00:00Z"]),
        "source_valid": [True], "hyg_return": [0.01], "lqd_return": [0.02],
        "relative_credit_return": [-0.01], "btc_variation": [0.2], "btc_variation_rank": [0.7],
    })
    assert support.build_clock(states).side.tolist() == [-1]
    assert support.build_clock(states, "hyg_only").side.tolist() == [1]
    assert support.build_clock(states, "inverse_lqd_only").side.tolist() == [-1]
