import numpy as np
import pandas as pd
from training import build_high_volatility_quartile_median_staircase_relay_support as s
def test_staircase_statistics_detects_four_ordered_medians():
 levels=np.repeat([100.,101.,102.,103.],120);block=pd.DataFrame({"open":np.repeat(100.,480),"high":levels,"low":np.minimum(100.,levels),"close":levels})
 m1,m2,m3,m4,side,migration,endpoint,variation=s.staircase_statistics(block);assert m1<m2<m3<m4 and side==1 and migration>0 and endpoint>0 and variation>0
def panel():
 return pd.DataFrame({"source_valid":[True]*6,"median_1":[1.]*6,"median_2":[2.]*6,"median_3":[3.]*6,"median_4":[4.]*6,"staircase_side":[1,1,-1,0,-1,-1],"migration_span":[.1]*6,"migration_rank":[.5,.8,.9,.4,.8,.9],"endpoint_displacement":[.1,-.3,-.4,-.2,.5,.4],"endpoint_rank":[.5,.8,.9,.4,.8,.9],"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01T01:00:00Z",periods=6,freq="8h")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,False,False,True,False] and side[active].tolist()==[1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 endpoint_active,endpoint_side,_=s.active(panel(),"endpoint_displacement");assert endpoint_active.tolist()==[False,True,False,False,True,False] and endpoint_side[endpoint_active].tolist()==[-1,1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
