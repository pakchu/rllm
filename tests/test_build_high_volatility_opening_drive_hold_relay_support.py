import numpy as np
import pandas as pd
from training import build_high_volatility_opening_drive_hold_relay_support as s
def test_upward_opening_drive_midpoint_hold_and_extension_is_long():
 opening=pd.DataFrame({"open":np.r_[90.,np.repeat(95.,59)],"high":np.r_[100.,np.repeat(99.,59)],"low":np.repeat(89.,60),"close":np.repeat(96.,60)})
 acceptance=pd.DataFrame({"close":np.r_[np.repeat(95.,419),101.]})
 high,low,mid,oo,oc,direction,minimum,maximum,terminal,side,no_midpoint,terminal_break=s.opening_drive_pattern(opening,acceptance)
 assert (high,low,mid,direction,minimum,terminal,side,no_midpoint,terminal_break)==(100.,89.,94.5,1,95.,101.,1,1,1)
def panel():
 return pd.DataFrame({"source_valid":[True]*6,"accepted_side":[0,1,1,0,-1,-1],"no_midpoint_side":[1,1,1,-1,-1,-1],"terminal_break_side":[-1,1,1,-1,-1,-1],"stale_opening_range_side":[0,-1,-1,0,1,1],"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01T02:00:00Z",periods=6,freq="8h")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,True,False,True,True] and side[active].tolist()==[1,1,-1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 stale_active,stale_side,_=s.active(panel(),"one_block_stale_opening_range");assert stale_active.tolist()==[False,True,True,False,True,True] and stale_side[stale_active].tolist()==[-1,-1,1,1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
