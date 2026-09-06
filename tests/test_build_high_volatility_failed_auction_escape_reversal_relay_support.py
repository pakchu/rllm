import numpy as np
import pandas as pd
from training import build_high_volatility_failed_auction_escape_reversal_relay_support as s
def test_upper_failed_escape_then_bearish_midpoint_confirmation_is_short():
 escape=pd.DataFrame({"high":np.r_[101.,np.repeat(99.,119)],"low":np.repeat(91.,120),"close":np.repeat(96.,120)})
 confirmation=pd.DataFrame({"open":np.r_[96.,np.repeat(94.,119)],"close":np.repeat(94.,120)})
 eh,el,ec,failed,co,cc,side,no_failed,reclaim=s.failed_escape_pattern(escape,confirmation,100.,90.)
 assert eh==101 and el==91 and ec==96 and failed==-1 and side==-1 and no_failed==-1 and reclaim==-1
def panel():
 return pd.DataFrame({"source_valid":[True]*6,"accepted_side":[0,1,1,0,-1,-1],"no_failed_escape_side":[1,1,1,-1,-1,-1],"reclaim_close_side":[-1,1,1,-1,-1,-1],"stale_prior_range_side":[0,-1,-1,0,1,1],"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01T02:00:00Z",periods=6,freq="8h")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,True,False,True,True] and side[active].tolist()==[1,1,-1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 stale_active,stale_side,_=s.active(panel(),"one_block_stale_prior_range");assert stale_active.tolist()==[False,True,True,False,True,True] and stale_side[stale_active].tolist()==[-1,-1,1,1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
