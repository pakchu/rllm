import numpy as np
import pandas as pd
from training import build_high_volatility_price_level_occupation_relay_support as s

def test_occupation_statistics():
 open_=np.repeat(100.,480);close=np.r_[np.repeat(101.,300),np.repeat(99.,180)]
 block=pd.DataFrame({"open":open_,"high":np.maximum(open_,close),"low":np.minimum(open_,close),"close":close})
 reference,above,below,occupation,median,variation=s.occupation_statistics(block)
 assert reference==100 and above==300 and below==180 and occupation>0 and median>0 and variation>0

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"reference_level":[100.]*6,"above_count":[300]*6,"below_count":[180]*6,"level_occupation":[-.1,.3,.4,.2,-.5,-.4],"occupation_rank":[.5,.8,.9,.4,.8,.9],"median_level_displacement":[.1,-.3,-.4,-.2,.5,.4],"median_displacement_rank":[.5,.8,.9,.4,.8,.9],"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01T03:00:00Z",periods=6,freq="8h")})

def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,False,False,True,False] and side[active].tolist()==[1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 median_active,median_side,_=s.active(panel(),"median_level_displacement");assert median_active.tolist()==[False,True,False,False,True,False] and median_side[median_active].tolist()==[-1,1]
 forced_active,forced_side,_=s.active(panel(),"same_clock_forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()

def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
