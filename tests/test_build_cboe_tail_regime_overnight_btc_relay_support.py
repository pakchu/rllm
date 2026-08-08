import pandas as pd
from training import build_cboe_tail_regime_overnight_btc_relay_support as support


def frame(term=.8, btc=-.02):
    return pd.DataFrame({"observation_date":[pd.Timestamp("2024-06-03").date()],"next_source_date":[pd.Timestamp("2024-06-04").date()],"overnight_start_time":[pd.Timestamp("2024-06-03T20:00:00Z")],"decision_time":[pd.Timestamp("2024-06-04T14:00:00Z")],"skew_rank":[term],"relative_convexity_rank":[term],"tail_pressure":[term],"overnight_btc_return":[btc],"valid":[True]})


def test_cvtlr_requires_tail_regime_confirmation_and_enters_five_minutes_later():
    clock=support.build_clock(frame());assert len(clock)==1 and clock.iloc[0].side==-1
    assert clock.iloc[0].entry_time==pd.Timestamp("2024-06-04T14:05:00Z") and clock.iloc[0].exit_time==pd.Timestamp("2024-06-04T20:05:00Z")


def test_cvtlr_rejects_neutral_or_unconfirmed_state():
    assert support.build_clock(frame(term=.50)).empty
    assert support.build_clock(frame(btc=.02)).empty
    assert len(support.build_clock(frame(term=.50),"no_tail_gate"))==1
