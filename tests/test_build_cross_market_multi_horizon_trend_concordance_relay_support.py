import numpy as np
import pandas as pd
from training import build_cross_market_multi_horizon_trend_concordance_relay_support as support

def frame():
 return pd.DataFrame({"session_date":["2024-01-01"]*4,"decision_time":pd.date_range("2024-01-01T02:00:00Z",periods=4,freq="1D"),"btc_6h_return":[.01,-.01,.01,.01],"btc_24h_return":[.02,-.02,-.02,.02],"eth_6h_return":[.03,-.03,.03,.03],"trend_sign":[1,-1,0,1],"concordant":[True,True,False,True],"btc_realized_variation":[.1]*4,"variation_rank":[.8,.5,.8,.8]})
def test_rank_is_strict_prior():
 r=support.strict_prior_midrank(pd.Series([1.,2.,3.]),lookback=2,minimum=2);assert np.isnan(r.iloc[:2]).all();assert r.iloc[2]==1.
def test_primary_and_controls():
 f=frame();a,s,_=support.conditions(f);assert a.tolist()==[True,False,False,True];assert s.tolist()==[1,-1,0,1];assert support.conditions(f,"no_volatility_gate")[0].tolist()==[True,True,False,True];assert support.conditions(f,"btc_horizons_only")[0].tolist()==[True,False,False,True];assert support.conditions(f,"btc_6h_eth_6h_only")[0].tolist()==[True,False,True,True];assert support.conditions(f,"direction_flip")[1].tolist()==[-1,1,0,-1];assert support.conditions(f,"forced_long")[1].tolist()==[1,1,1,1]
def test_clock_delays_five_minutes_holds_24h_and_reserves():
 c=support.build_clock(frame());assert len(c)==2;assert c.iloc[0].entry_time==pd.Timestamp("2024-01-01T02:05:00Z");assert c.iloc[0].exit_time==pd.Timestamp("2024-01-02T02:05:00Z")
def test_builder_binding_and_sealed_outcomes():
 assert support.sha(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA;source=support.BUILDER.read_text();assert '"postentry_return_pnl_execution_price_opened": False' in source;assert '"gross9_rows_opened": False' in source;assert support.ECONOMIC_OUTCOMES_AUTHORIZED is False
