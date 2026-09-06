import numpy as np
import pandas as pd
from training import build_high_volatility_cci_extreme_reentry_relay_support as s

def test_cci_uses_same_window_mean_deviation():
 old=s.P["cci_periods"];s.P["cci_periods"]=3
 try:
  h=pd.Series([3.,6.,9.]);l=pd.Series([1.,4.,7.]);c=pd.Series([2.,5.,8.]);v=pd.Series([True]*3);x=s.cci(h,l,c,v)
  assert x.center.iloc[2]==5 and x.mean_deviation.iloc[2]==2
  assert np.isclose(x.cci.iloc[2],3/(.015*2))
 finally:s.P["cci_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"cci":[-120.,-110.,-90.,0.,120.,90.,0.],"variation_rank":[.8,.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[5]
 outward,side,_=s.active(panel(),"outward_crossing");assert outward.iloc[4] and side.iloc[4]==1
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[3] and side.iloc[3]==1
