import numpy as np
import pandas as pd
from training import build_high_volatility_relative_vigor_crossover_relay_support as s

def test_weighted_four_is_1221_current_to_oldest():
 x=pd.Series([1.,2.,3.,4.]);assert s.weighted_four(x).iloc[3]==(4+2*3+2*2+1)/6

def test_relative_vigor_common_smoothing():
 old=s.P["rvi_periods"];s.P["rvi_periods"]=2
 try:
  o=pd.Series([1.,2.,3.,4.,5.,6.]);c=o+1;h=c+1;l=o-1;v=pd.Series([True]*6);x=s.relative_vigor(o,h,l,c,v)
  assert np.isclose(x.rvi.iloc[4],1/3) and np.isclose(x.raw_rvi.iloc[4],1/3)
 finally:s.P["rvi_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*6,"difference":[-.2,-.1,.1,.2,-.1,-.2],"raw_rvi":[-.2,-.1,.1,.2,-.1,-.2],"raw_signal":[0.]*6,"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_crossover");assert stale.iloc[3] and side.iloc[3]==1
