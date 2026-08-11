import numpy as np
import pandas as pd
from training import build_high_volatility_chande_kroll_stop_breakout_relay_support as s

def test_chande_kroll_wilder_stops_are_finite_after_warmup():
 old=s.P.copy();s.P.update({"atr_periods":2,"stop_periods":2,"atr_multiplier":1.0})
 try:
  frame=s.chande_kroll(pd.Series([2.,3.,4.,8.,9.]),pd.Series([1.,2.,3.,7.,8.]),pd.Series([1.5,2.5,3.5,7.5,8.5]),pd.Series([True]*5));assert np.isfinite(frame.atr.iloc[2]) and np.isfinite(frame.short_stop.iloc[3]) and np.isfinite(frame.long_stop.iloc[3])
 finally:s.P.clear();s.P.update(old)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"preliminary_side":[-1,-1,1,1,-1,-1],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
