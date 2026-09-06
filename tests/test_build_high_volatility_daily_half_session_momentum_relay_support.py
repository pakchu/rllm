import numpy as np
import pandas as pd

from training import build_high_volatility_daily_half_session_momentum_relay_support as support


def panel():
    return pd.DataFrame({
        "source_day": pd.date_range("2024-01-01", periods=4, freq="1d", tz="UTC"),
        "feature_available_time": pd.date_range("2024-01-01 12:00", periods=4, freq="1d", tz="UTC"),
        "source_valid": [True] * 4,
        "first_half_return": [0.1, -0.1, 0.2, -0.2],
        "late_six_hour_return": [-0.1, -0.1, 0.1, 0.1],
        "realized_variation": [0.2] * 4,
        "variation_rank": [0.8, 0.4, 0.9, 0.7],
        "eligible": [True, False, True, True],
    })


def test_prior_rank_excludes_current():
    ranks=support.prior_rank(pd.Series(np.arange(181,dtype=float)))
    assert ranks.iloc[:180].isna().all() and ranks.iloc[180]==1.0


def test_primary_and_controls_are_frozen():
    activity,side,_=support.active(panel())
    assert activity.tolist()==[True,False,True,True]
    assert side.tolist()==[1,-1,1,-1]
    assert support.active(panel(),"no_variation_gate")[0].tolist()==[True]*4
    assert support.active(panel(),"late_six_hour_direction")[1].tolist()==[-1,-1,1,1]
    assert support.active(panel(),"direction_flip")[1].tolist()==[-1,1,-1,1]


def test_clock_delays_five_minutes_and_holds_twelve_hours():
    clock=support.build_clock(panel())
    assert clock.entry_time.iloc[0]==pd.Timestamp("2024-01-01 12:05",tz="UTC")
    assert clock.exit_time.iloc[0]==pd.Timestamp("2024-01-02 00:05",tz="UTC")
    assert clock.feature_available_time.iloc[0] < clock.entry_time.iloc[0]


def test_binding_and_sealed_outcomes():
    assert support.sha(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA
    source=support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
