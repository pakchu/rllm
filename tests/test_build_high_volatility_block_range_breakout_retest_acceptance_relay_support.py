import numpy as np
import pandas as pd
from training import build_high_volatility_block_range_breakout_retest_acceptance_relay_support as s
def test_upper_break_retest_acceptance_is_ordered():
 close=np.full(240,99.);high=np.full(240,99.5);low=np.full(240,98.5);high[20]=101.;low[21]=100.;close[-1]=100.5
 confirmation=pd.DataFrame({"high":high,"low":low,"close":close});upper,lower,side,no_retest=s.accepted_pattern(confirmation,100.,90.);assert upper and not lower and side==1 and no_retest==1
def panel():
 return pd.DataFrame({"source_valid":[True]*6,"prior_high":[100.]*6,"prior_low":[90.]*6,"upper_pattern":[False,True,True,False,False,False],"lower_pattern":[False,False,False,False,True,True],"accepted_side":[0,1,1,0,-1,-1],"no_retest_side":[1,1,1,-1,-1,-1],"stale_prior_range_side":[0,-1,-1,0,1,1],"variation_rank":[.8,.8,.8,.8,.8,.8],"feature_available_time":pd.date_range("2024-01-01T00:00:00Z",periods=6,freq="8h")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,True,False,True,True] and side[active].tolist()==[1,1,-1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 stale_active,stale_side,_=s.active(panel(),"one_block_stale_prior_range");assert stale_active.tolist()==[False,True,True,False,True,True] and stale_side[stale_active].tolist()==[-1,-1,1,1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
