import numpy as np
import pandas as pd
from training import build_high_volatility_random_walk_index_cross_relay_support as s

def test_cross_side():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 bars=pd.DataFrame({"bar_high":[2.,3.,5.,7.],"bar_low":[1.,2.,3.,4.],"bar_close":[1.5,2.5,4.,6.]})
 values=s.rwi_values(bars,pd.Series([True]*4),2);assert np.isnan(values.rwi_high.iloc[2]) and values.rwi_high.iloc[3]>values.rwi_low.iloc[3]

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"fixed_14_side":[0,0,1,0,-1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
