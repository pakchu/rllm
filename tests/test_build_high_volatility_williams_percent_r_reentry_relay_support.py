import numpy as np
import pandas as pd
from training import build_high_volatility_williams_percent_r_reentry_relay_support as s

def test_percent_r_range_location():
 old=s.P["percent_r_periods"];s.P["percent_r_periods"]=2
 try:
  h=pd.Series([10.,12.,11.]);l=pd.Series([6.,8.,7.]);c=pd.Series([8.,9.,10.]);v=pd.Series([True]*3);x=s.percent_r(h,l,c,v);assert x.highest_high.iloc[1]==12 and x.lowest_low.iloc[1]==6 and x.percent_r.iloc[1]==-50 and x.percent_r.iloc[2]==-40
 finally:s.P["percent_r_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"percent_r":[-90.,-85.,-70.,-50.,-10.,-25.,-50.],"variation_rank":[.8,.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[5]
 outward,side,_=s.active(panel(),"outward_crossing");assert outward.iloc[4] and side.iloc[4]==1
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[3] and side.iloc[3]==1
