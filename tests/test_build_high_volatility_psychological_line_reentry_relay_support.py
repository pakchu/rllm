import numpy as np
import pandas as pd
from training import build_high_volatility_psychological_line_reentry_relay_support as s

def test_psy_counts_strict_up_and_ties_as_zero():
 old=s.P["psy_periods"];s.P["psy_periods"]=3
 try:
  x=s.psychological_line(pd.Series([1.,2.,2.,1.,3.]),pd.Series([True]*5));assert np.isnan(x.psy.iloc[2]) and np.isclose(x.psy.iloc[3],100/3) and np.isclose(x.psy.iloc[4],100/3)
 finally:s.P["psy_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"psy":[20.,25.,40.,50.,80.,70.,50.],"variation_rank":[.8,.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[5]
 outward,side,_=s.active(panel(),"outward_break");assert outward.iloc[4] and side.iloc[4]==1
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[3] and side.iloc[3]==1
