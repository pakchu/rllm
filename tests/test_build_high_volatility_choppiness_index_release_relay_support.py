import numpy as np
import pandas as pd
from training import build_high_volatility_choppiness_index_release_relay_support as s

def test_choppiness_canonical_ratio():
 old=s.P["choppiness_periods"];s.P["choppiness_periods"]=3
 try:
  high=pd.Series([2.,3.,4.,5.]);low=pd.Series([1.,2.,3.,4.]);close=pd.Series([1.5,2.5,3.5,4.5]);valid=pd.Series([True]*4)
  frame=s.choppiness(high,low,close,valid)
  assert np.isclose(frame.choppiness.iloc[3],100*np.log10(4.5/3)/np.log10(3))
 finally:s.P["choppiness_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"choppiness":[45.,40.,37.,36.,40.,37.],"direction":[-1,-1,1,1,-1,-1],"variation_rank":[.8,.8,.8,.8,.8,.4],"bar_high":[2.]*6,"bar_low":[1.]*6,"bar_close":[1.6]*6})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[5]
 stale,side,_=s.active(panel(),"one_bar_stale_release");assert stale.iloc[3] and side.iloc[3]==1
