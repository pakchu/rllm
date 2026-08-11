import numpy as np
import pandas as pd
from training import build_high_volatility_relative_vigor_index_crossover_relay_support as s

def test_symmetric_smoothing_and_rvi():
 old=s.P["rvi_periods"];s.P["rvi_periods"]=2
 try:
  o=pd.Series(np.arange(1,10,dtype=float));c=o+pd.Series(np.arange(1,10,dtype=float));h=c+1;l=o-1;v=pd.Series([True]*9)
  x=s.relative_vigor(o,h,l,c,v)
  assert np.isclose(x.body_smooth.iloc[3],15/6) and np.isclose(x.range_smooth.iloc[3],27/6)
  assert np.isfinite(x.rvi.iloc[4]) and np.isnan(x.signal.iloc[6]) and np.isfinite(x.signal.iloc[7])
 finally:s.P["rvi_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*6,"rvi":[-.2,-.1,.1,.2,-.1,-.2],"signal":[-.1,0.,0.,.1,.0,-.1],"difference":[-.1,-.1,.1,.1,-.1,-.1],"raw_ratio":[-.2,-.1,.1,.2,-.1,-.2],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[2] and z.iloc[2]==1 and not a.iloc[4]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_crossover");assert stale.iloc[3] and side.iloc[3]==1
