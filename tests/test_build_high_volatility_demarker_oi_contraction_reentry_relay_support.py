import numpy as np
import pandas as pd
from training import build_high_volatility_demarker_oi_contraction_reentry_relay_support as s

def test_demarker_uses_positive_high_and_low_expansion():
 old=s.P["demarker_periods"];s.P["demarker_periods"]=2
 try:
  h=pd.Series([10.,12.,11.]);l=pd.Series([8.,9.,7.]);v=pd.Series([True]*3);x=s.demarker(h,l,v)
  assert x.demax.iloc[1]==2 and x.demin.iloc[1]==0 and x.demax.iloc[2]==0 and x.demin.iloc[2]==2
  assert x.demarker.iloc[2]==.5
 finally:s.P["demarker_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"demarker":[.2,.25,.4,.5,.8,.65,.5],"variation_rank":[.8,.8,.8,.8,.8,.4,.8],"oi_change":[-.1]*7})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[5]
 outward,side,_=s.active(panel(),"outward_crossing");assert outward.iloc[4] and side.iloc[4]==1
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[3] and side.iloc[3]==1
