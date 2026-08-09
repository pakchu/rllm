import numpy as np
import pandas as pd
from training import build_cross_alt_breadth_underreaction_relay_support as support

def frame():
    return pd.DataFrame({"session_date":["2024-01-01"]*4,"decision_time":pd.date_range("2024-01-01T04:00:00Z",periods=4,freq="8h"),"btc_return":[.01,-.01,.03,.01],"confirming_alts":[5,4,5,3],"alt_majority_sign":[1,-1,1,-1],"alt_impulse":[.02]*4,"impulse_rank":[.8]*4,"underreaction_ratio":[.5,.5,1.5,.5],"btc_realized_variation":[.1]*4,"variation_rank":[.8,.5,.8,.8]})

def test_strict_prior_rank_excludes_current_and_caps_history():
    rank=support.strict_prior_midrank(pd.Series([1.,2.,3.,2.]),lookback=2,minimum=2);assert np.isnan(rank.iloc[:2]).all();assert rank.iloc[2]==1.;assert rank.iloc[3]==.25

def test_primary_and_frozen_controls():
    f=frame();active,side,_=support.conditions(f);assert active.tolist()==[True,False,False,False];assert side.tolist()==[1,-1,1,-1];assert support.conditions(f,"no_volatility_gate")[0].tolist()==[True,True,False,False];assert support.conditions(f,"no_underreaction_gate")[0].tolist()==[True,False,True,False];assert support.conditions(f,"three_of_six_breadth")[0].tolist()==[True,False,False,True];assert support.conditions(f,"direction_flip")[1].tolist()==[-1,1,-1,1];assert support.conditions(f,"forced_long")[1].tolist()==[1,1,1,1]

def test_clock_uses_five_minute_delay_and_holds_eight_hours():
    c=support.build_clock(frame());assert len(c)==1;assert c.iloc[0].entry_time==pd.Timestamp("2024-01-01T04:05:00Z");assert c.iloc[0].exit_time==pd.Timestamp("2024-01-01T12:05:00Z")

def test_builder_is_bound_and_outcomes_are_sealed():
    source=support.BUILDER.read_text();assert support.PREREG_SHA=="8d51df306ceab0c90069c09dbb8c123f1705f80856748d48df7ff41fecaed621";assert "FROM bars_binance" in source;assert '"postentry_return_pnl_execution_price_opened": False' in source;assert '"gross9_rows_opened": False' in source;assert support.ECONOMIC_OUTCOMES_AUTHORIZED is False
