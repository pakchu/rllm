import numpy as np
import pandas as pd
from training import build_high_volatility_relative_momentum_index_reentry_relay_support as s

def test_rmi_uses_five_period_momentum_and_wilder_seed():
 old=dict(s.P);s.P.update(momentum_periods=2,rmi_periods=3)
 try:
  x=s.relative_momentum_index(pd.Series([10.,11.,12.,14.,13.,16.]),pd.Series([True]*6));assert x.momentum.iloc[2]==2 and x.momentum.iloc[4]==1 and np.isnan(x.rmi.iloc[3]) and 0<x.rmi.iloc[4]<=100
 finally:s.P.clear();s.P.update(old)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"rmi":[20.,25.,40.,50.,80.,65.,50.],"variation_rank":[.8,.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[5]
 outward,side,_=s.active(panel(),"outward_crossing");assert outward.iloc[4] and side.iloc[4]==1
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[3] and side.iloc[3]==1
