import pandas as pd
from training import build_cboe_volatility_hour_momentum_relay_support as support


def frame(relative=.8, btc=-.02, threshold=.01):
    return pd.DataFrame({"observation_date":[pd.Timestamp("2024-06-03").date()],"next_source_date":[pd.Timestamp("2024-06-04").date()],"decision_hour_start":[pd.Timestamp("2024-06-04T13:00:00Z")],"decision_time":[pd.Timestamp("2024-06-04T14:00:00Z")],"relative_convexity_rank":[relative],"previous_relative_convexity_rank":[relative],"vix_level_rank":[relative],"hour_return":[btc],"prior_abs_hour_q75":[threshold],"valid":[True]})


def test_cvhmr_requires_volatility_backed_overnight_tail_and_enters_five_minutes_later():
    clock=support.build_clock(frame());assert len(clock)==1 and clock.iloc[0].side==-1
    assert clock.iloc[0].entry_time==pd.Timestamp("2024-06-04T14:05:00Z") and clock.iloc[0].exit_time==pd.Timestamp("2024-06-04T20:05:00Z")


def test_cvhmr_rejects_low_volatility_or_small_overnight_move():
    assert support.build_clock(frame(relative=.50)).empty
    assert support.build_clock(frame(btc=.005)).empty
    assert len(support.build_clock(frame(relative=.50),"no_relative_convexity_gate"))==1
