import pandas as pd
from training import build_volatility_conditioned_overnight_momentum_relay_support as support


def frame(relative=.8, tail=.8, btc=-.02):
    return pd.DataFrame({"observation_date":[pd.Timestamp("2024-06-03").date()],"next_source_date":[pd.Timestamp("2024-06-04").date()],"overnight_start_time":[pd.Timestamp("2024-06-03T20:00:00Z")],"decision_time":[pd.Timestamp("2024-06-04T14:00:00Z")],"relative_convexity_rank":[relative],"previous_relative_convexity_rank":[relative],"vix_level_rank":[relative],"overnight_abs_rank":[tail],"overnight_btc_return":[btc],"valid":[True]})


def test_vomr_requires_volatility_backed_overnight_tail_and_enters_five_minutes_later():
    clock=support.build_clock(frame());assert len(clock)==1 and clock.iloc[0].side==-1
    assert clock.iloc[0].entry_time==pd.Timestamp("2024-06-04T14:05:00Z") and clock.iloc[0].exit_time==pd.Timestamp("2024-06-04T20:05:00Z")


def test_vomr_rejects_low_volatility_or_small_overnight_move():
    assert support.build_clock(frame(relative=.50)).empty
    assert support.build_clock(frame(tail=.74)).empty
    assert len(support.build_clock(frame(relative=.50),"no_relative_convexity_gate"))==1
