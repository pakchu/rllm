import numpy as np
import pandas as pd
from training import build_high_volatility_ichimoku_tenkan_kijun_crossover_relay_support as s

def test_tenkan_and_kijun_midpoints():
 old=(s.P["tenkan_periods"],s.P["kijun_periods"]);s.P["tenkan_periods"]=2;s.P["kijun_periods"]=3
 try:
  h=pd.Series([10.,12.,11.,15.]);l=pd.Series([6.,4.,9.,12.]);v=pd.Series([True]*4);x=s.ichimoku(h,l,v);assert x.tenkan.iloc[1]==8 and x.kijun.iloc[2]==8 and x.difference.iloc[3]!=0
 finally:s.P["tenkan_periods"],s.P["kijun_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*6,"difference":[-2.,-1.,1.,2.,-1.,-2.],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 assert s.active(panel(),"persistent_equilibrium_state")[0].iloc[3]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
