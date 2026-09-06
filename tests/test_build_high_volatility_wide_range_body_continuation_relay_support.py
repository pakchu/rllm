import numpy as np
import pandas as pd
from training import build_high_volatility_wide_range_body_continuation_relay_support as s
def test_wide_efficient_up_body_is_long():
 current=pd.DataFrame({"open":np.r_[90.,np.repeat(95.,479)],"high":np.repeat(110.,480),"low":np.repeat(89.,480),"close":np.repeat(105.,480)})
 high,low,width,opening,close,efficiency,body_side,side,no_wide=s.wide_range_pattern(current,[10.,15.,20.])
 assert (high,low,width,opening,close,body_side,side,no_wide)==(110.,89.,21.,90.,105.,1,1,1) and efficiency>.5
def panel():
 return pd.DataFrame({"source_valid":[True]*6,"accepted_side":[0,1,1,0,-1,-1],"no_wide_side":[1,1,1,-1,-1,-1],"body_side":[-1,1,1,-1,-1,-1],"stale_prior_ranges_side":[0,-1,-1,0,1,1],"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01T02:00:00Z",periods=6,freq="8h")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,True,False,True,True] and side[active].tolist()==[1,1,-1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 stale_active,stale_side,_=s.active(panel(),"one_block_stale_prior_ranges");assert stale_active.tolist()==[False,True,True,False,True,True] and stale_side[stale_active].tolist()==[-1,-1,1,1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
