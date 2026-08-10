import numpy as np
import pandas as pd
from training import build_high_volatility_median_crossing_impulse_asymmetry_relay_support as s

def test_crossing_statistics_positive_upward_impulse():
 path=np.zeros(480);path[:5]=[-1.,3.,-.2,3.,-.2];open_=np.repeat(100.,480);close=100*np.exp(path);block=pd.DataFrame({"open":open_,"high":np.maximum(open_,close),"low":np.minimum(open_,close),"close":close})
 median,ups,downs,up_mass,down_mass,asymmetry,step_asymmetry,variation=s.crossing_statistics(block)
 assert np.isclose(median,np.log(100.)) and ups==2 and downs==2 and up_mass>down_mass and asymmetry>0 and step_asymmetry>0 and variation>0

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"median_level":[0.]*6,"upcross_count":[2]*6,"downcross_count":[2]*6,"up_overshoot_mass":[2.]*6,"down_overshoot_mass":[1.]*6,"crossing_impulse_asymmetry":[-.1,.3,.4,.2,-.5,-.4],"asymmetry_rank":[.5,.8,.9,.4,.8,.9],"crossing_step_return_asymmetry":[.1,-.3,-.4,-.2,.5,.4],"step_asymmetry_rank":[.5,.8,.9,.4,.8,.9],"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01T07:00:00Z",periods=6,freq="8h")})

def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,False,False,True,False] and side[active].tolist()==[1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 step_active,step_side,_=s.active(panel(),"crossing_step_return_asymmetry");assert step_active.tolist()==[False,True,False,False,True,False] and step_side[step_active].tolist()==[-1,1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()

def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
