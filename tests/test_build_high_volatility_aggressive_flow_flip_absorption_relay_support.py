import numpy as np
import pandas as pd
from training import build_high_volatility_aggressive_flow_flip_absorption_relay_support as s
def test_up_price_persistence_with_supporting_to_opposing_flow_is_long():
 first=pd.DataFrame({"open":np.r_[100.,np.repeat(105.,239)],"close":np.repeat(110.,240),"quote_asset_volume":np.repeat(100.,240),"taker_buy_quote":np.repeat(60.,240)})
 second=pd.DataFrame({"open":np.r_[110.,np.repeat(115.,239)],"close":np.repeat(120.,240),"quote_asset_volume":np.repeat(100.,240),"taker_buy_quote":np.repeat(40.,240)})
 r1,r2,price,f1,f2,ordered,side,no_flow,transition=s.flow_flip_pattern(first,second)
 assert r1>0 and r2>0 and price==1 and f1>0 and f2<0 and ordered and side==1 and no_flow==1 and transition==1
def panel():
 return pd.DataFrame({"source_valid":[True]*6,"accepted_side":[0,1,1,0,-1,-1],"no_flow_flip_side":[1,1,1,-1,-1,-1],"flow_transition_side":[-1,1,1,-1,-1,-1],"stale_flow_side":[0,-1,-1,0,1,1],"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01T02:00:00Z",periods=6,freq="8h")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,True,False,True,True] and side[active].tolist()==[1,1,-1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 stale_active,stale_side,_=s.active(panel(),"one_block_stale_flow");assert stale_active.tolist()==[False,True,True,False,True,True] and stale_side[stale_active].tolist()==[-1,-1,1,1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
