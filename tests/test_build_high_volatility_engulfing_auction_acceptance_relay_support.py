import numpy as np
import pandas as pd
from training import build_high_volatility_engulfing_auction_acceptance_relay_support as s
def test_engulfing_pattern_accepts_upper_quartile_bullish_close():
 auction=pd.DataFrame({"open":np.r_[95.,np.repeat(100.,239)],"high":np.r_[111.,np.repeat(105.,239)],"low":np.r_[89.,np.repeat(95.,239)],"close":np.r_[100.,np.repeat(109.,239)]})
 eo,eh,el,ec,location,engulf,side,no_outer,body=s.engulfing_pattern(auction,110.,90.);assert engulf and location>=.75 and side==1 and no_outer==1 and body==1
def panel():
 return pd.DataFrame({"source_valid":[True]*6,"accepted_side":[0,1,1,0,-1,-1],"no_outer_quartile_side":[1,1,1,-1,-1,-1],"body_only_side":[1,-1,1,-1,1,-1],"stale_prior_range_side":[0,-1,-1,0,1,1],"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01T04:00:00Z",periods=6,freq="8h")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,True,False,True,True] and side[active].tolist()==[1,1,-1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 stale_active,stale_side,_=s.active(panel(),"one_block_stale_prior_range");assert stale_active.tolist()==[False,True,True,False,True,True] and stale_side[stale_active].tolist()==[-1,-1,1,1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
