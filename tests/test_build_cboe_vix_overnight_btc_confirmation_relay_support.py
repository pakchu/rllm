import pandas as pd
from training import build_cboe_vix_overnight_btc_confirmation_relay_support as support


def frame(dv=.1, btc=-.02, rank=.8):
    return pd.DataFrame({"observation_date":[pd.Timestamp("2024-06-03").date()],"next_source_date":[pd.Timestamp("2024-06-04").date()],"overnight_start_time":[pd.Timestamp("2024-06-03T20:00:00Z")],"decision_time":[pd.Timestamp("2024-06-04T14:00:00Z")],"delta_log_vix":[dv],"absolute_vix_change_rank":[rank],"overnight_btc_return":[btc],"valid":[True]})


def test_cvobr_requires_cross_market_confirmation_and_enters_five_minutes_later():
    clock=support.build_clock(frame());assert len(clock)==1 and clock.iloc[0].side==-1
    assert clock.iloc[0].entry_time==pd.Timestamp("2024-06-04T14:05:00Z") and clock.iloc[0].exit_time==pd.Timestamp("2024-06-04T20:05:00Z")


def test_cvobr_rejects_small_or_unconfirmed_shock():
    assert support.build_clock(frame(rank=.74)).empty
    assert support.build_clock(frame(btc=.02)).empty
    assert len(support.build_clock(frame(rank=.74),"no_extreme_gate"))==1
