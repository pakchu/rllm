import numpy as np
import pandas as pd
from training import build_high_volatility_schaff_trend_cycle_reentry_relay_support as s

def test_stochastic_and_threshold_side():
 old=s.P.copy();s.P.update({"lower_reentry":25.,"upper_reentry":75.})
 try:
  values=pd.Series([0.,1.,2.]);result=s.stochastic(values,pd.Series([True]*3),3);assert result.iloc[2]==100
  assert s.threshold_side(pd.Series([20.,30.,80.,70.])).tolist()==[0,1,0,-1]
 finally:s.P.clear();s.P.update(old)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"macd_side":[0,0,1,0,-1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[3] and side.iloc[3]==1
