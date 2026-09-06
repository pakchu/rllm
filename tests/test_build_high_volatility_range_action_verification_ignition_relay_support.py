import numpy as np
import pandas as pd
from training import build_high_volatility_range_action_verification_ignition_relay_support as s

def test_ravi_is_signed_simple_average_separation():
 old_fast,old_slow=s.P["fast_periods"],s.P["slow_periods"];s.P["fast_periods"],s.P["slow_periods"]=2,4
 try:
  x=s.ravi(pd.Series([10.,10.,10.,14.]),pd.Series([True]*4));assert x.fast_sma.iloc[3]==12 and x.slow_sma.iloc[3]==11 and np.isclose(x.signed_spread_pct.iloc[3],100/11) and x.ravi.iloc[3]>0
 finally:s.P["fast_periods"],s.P["slow_periods"]=old_fast,old_slow

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"ravi":[2.,2.5,3.5,4.,3.5,2.5,2.],"entry_side":[-1,-1,1,1,-1,-1,-1],"variation_rank":[.8]*7})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[2]
 release,side,_=s.active(panel(),"trend_release");assert bool(release.iloc[5]) and side.iloc[5]==-1 and not bool(release.iloc[6])
 stale,side,_=s.active(panel(),"one_bar_stale_ignition");assert stale.iloc[3] and side.iloc[3]==1
 forced,side,_=s.active(panel(),"forced_long");assert side[forced].eq(1).all()
