import numpy as np
import pandas as pd
from training import build_high_volatility_hull_moving_average_turn_relay_support as s

def test_weighted_average_uses_recent_linear_weights():
 result=s.weighted_average(pd.Series([1.,2.,3.]),pd.Series([True]*3),3)
 assert np.isclose(result.iloc[2],14/6)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"slope":[-2.,-1.,1.,2.,-1.,-2.],"full_wma_slope":[-2.,-1.,1.,2.,-1.,-2.],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_turn");assert stale.iloc[3] and side.iloc[3]==1
