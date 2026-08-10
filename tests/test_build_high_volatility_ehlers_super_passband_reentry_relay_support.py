import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_super_passband_reentry_relay_support as s

def test_super_passband_seed_recursion_rms_and_reset():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 old=s.P["rms_period"];s.P["rms_period"]=3
 try:
  prices=pd.Series([10.,11.,13.,12.,14.,15.,99.,20.,21.,23.]);x=s.super_passband(prices,pd.Series([True]*6+[False]+[True]*3))
  assert x.passband.iloc[:2].tolist()==[0.,0.] and np.isfinite(x.passband.iloc[2]) and np.isfinite(x.rms.iloc[2])
  assert x.iloc[6][["passband","rms"]].isna().all() and x.run_length.iloc[7:9].tolist()==[1,2]
  assert x.passband.iloc[7:9].tolist()==[0.,0.] and np.isfinite(x.rms.iloc[9])
 finally:s.P["rms_period"]=old

def test_rms_reentry_direction():
 side=s.reentry_side(pd.Series([-2.,-.5,0.,2.,.5]),pd.Series([1.]*5));assert side.tolist()==[0,1,0,0,-1]

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"zero_cross_side":[0,-1,0,1,0,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 raw,side,_=s.active(panel(),"zero_cross");assert raw.iloc[1] and side.iloc[1]==-1 and raw.iloc[3] and side.iloc[3]==1
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[3] and side.iloc[3]==1
