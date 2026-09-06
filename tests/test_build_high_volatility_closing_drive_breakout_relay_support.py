import numpy as np
import pandas as pd
from training import build_high_volatility_closing_drive_breakout_relay_support as s
def test_inside_open_upper_closing_drive_is_long():
 balance=pd.DataFrame({"high":np.repeat(100.,420),"low":np.repeat(90.,420)})
 drive=pd.DataFrame({"open":np.r_[95.,np.repeat(100.,59)],"close":np.repeat(101.,60)})
 high,low,do,dc,inside,side,no_inside,body=s.closing_drive_pattern(balance,drive)
 assert (high,low,do,dc,inside,side,no_inside,body)==(100.,90.,95.,101.,True,1,1,1)
def panel():
 return pd.DataFrame({"source_valid":[True]*6,"accepted_side":[0,1,1,0,-1,-1],"no_inside_side":[1,1,1,-1,-1,-1],"drive_body_side":[-1,1,1,-1,-1,-1],"stale_balance_range_side":[0,-1,-1,0,1,1],"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01T02:00:00Z",periods=6,freq="8h")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,True,False,True,True] and side[active].tolist()==[1,1,-1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 stale_active,stale_side,_=s.active(panel(),"one_block_stale_balance_range");assert stale_active.tolist()==[False,True,True,False,True,True] and stale_side[stale_active].tolist()==[-1,-1,1,1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
