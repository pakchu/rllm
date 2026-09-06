import pandas as pd
from training import build_cboe_vix_acceleration_dual_confirmation_relay_support as support


def frame(acceleration=.1, overnight=-.02, opening=-.01):
    return pd.DataFrame({"observation_date":[pd.Timestamp("2024-06-03").date()],"next_source_date":[pd.Timestamp("2024-06-04").date()],"overnight_start_time":[pd.Timestamp("2024-06-03T20:00:00Z")],"decision_time":[pd.Timestamp("2024-06-04T14:00:00Z")],"delta_vix_level":[acceleration],"previous_delta_vix_level":[acceleration],"overnight_btc_return":[overnight],"opening_hour_return":[opening],"valid":[True]})


def test_cvvdmr_requires_dual_confirmation_and_enters_five_minutes_later():
    clock=support.build_clock(frame());assert len(clock)==1 and clock.iloc[0].side==-1
    assert clock.iloc[0].entry_time==pd.Timestamp("2024-06-04T14:05:00Z") and clock.iloc[0].exit_time==pd.Timestamp("2024-06-04T20:05:00Z")


def test_cvvdmr_rejects_nonaccelerating_or_disagreeing_paths():
    assert support.build_clock(frame(acceleration=-.1)).empty
    assert support.build_clock(frame(opening=.01)).empty
    assert len(support.build_clock(frame(acceleration=-.1),"no_vix_level_acceleration"))==1
