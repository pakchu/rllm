import numpy as np
import pandas as pd
from training import build_high_volatility_vortex_path_crossover_relay_support as s

def test_vortex_paths_and_common_normalizer():
 old=s.P["vortex_periods"];s.P["vortex_periods"]=2
 try:
  h=pd.Series([10.,12.,11.]);l=pd.Series([8.,9.,7.]);c=pd.Series([9.,11.,8.]);v=pd.Series([True]*3);x=s.vortex(h,l,c,v)
  assert x.positive_path.iloc[1]==4 and x.negative_path.iloc[1]==1
  assert x.positive_path.iloc[2]==2 and x.negative_path.iloc[2]==5
  assert np.isclose(x.vi_plus.iloc[2],6/7) and np.isclose(x.vi_minus.iloc[2],6/7)
 finally:s.P["vortex_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*6,"difference":[-.2,-.1,.1,.2,-.1,-.2],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_crossover");assert stale.iloc[3] and side.iloc[3]==1
