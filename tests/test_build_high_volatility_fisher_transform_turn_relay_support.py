import numpy as np
import pandas as pd
from training import build_high_volatility_fisher_transform_turn_relay_support as s

def test_fisher_transform_recurses_and_resets():
 old=s.P["range_periods"];s.P["range_periods"]=2
 try:
  h=pd.Series([2.,3.,4.,5.,6.]);l=h-2;v=pd.Series([True,True,True,False,True]);x=s.fisher_transform(h,l,v);assert np.isnan(x.fisher.iloc[0]) and np.isfinite(x.fisher.iloc[1]) and x.recursive_value.iloc[2]>x.recursive_value.iloc[1] and np.isnan(x.fisher.iloc[4])
 finally:s.P["range_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1
def panel():return pd.DataFrame({"source_valid":[True]*8,"fisher_slope":[0.,1.,2.,-1.,-2.,1.,2.,-1.],"raw_fisher_slope":[1.,1.,-1.,-2.,1.,2.,-1.,-2.],"variation_rank":[.8,.8,.8,.8,.8,.4,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[1] and z.iloc[1]==1 and a.iloc[3] and z.iloc[3]==-1 and not a.iloc[5] and a.iloc[7] and z.iloc[7]==-1
 assert s.active(panel(),"no_variation_gate")[0].iloc[5]
 raw,side,_=s.active(panel(),"unsmoothed_fisher");assert raw.iloc[2] and side.iloc[2]==-1 and raw.iloc[4] and side.iloc[4]==1
 stale,side,_=s.active(panel(),"one_bar_stale_turn");assert stale.iloc[2] and side.iloc[2]==1
