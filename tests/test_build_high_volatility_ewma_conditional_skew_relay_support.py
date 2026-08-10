import numpy as np
import pandas as pd
from training import build_high_volatility_ewma_conditional_skew_relay_support as s
def test_weighted_skew_is_finite_and_directional():
 values=np.r_[np.repeat(-.01,29),.20]
 assert s.weighted_skew(values)>0 and s.unweighted_skew(values)>0
def panel():
 return pd.DataFrame({"source_valid":[True]*6,"conditional_skew":[-1.,1.,1.,-1.,-1.,1.],"skew_strength_rank":[.5,.8,.8,.5,.8,.8],"unweighted_skew":[1.,1.,-1.,-1.,1.,-1.],"unweighted_skew_strength_rank":[.8]*6,"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01 02:00",periods=6,freq="1d",tz="UTC")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,True,False,True,True] and side[active].tolist()==[1,1,-1,1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 unweighted_active,unweighted_side,_=s.active(panel(),"unweighted_skew");assert unweighted_active.all() and unweighted_side.tolist()==[1,1,-1,-1,1,-1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(ranks.iloc[119]) and ranks.iloc[120]==1.
