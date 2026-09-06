import numpy as np
import pandas as pd
from training import build_high_volatility_twiggs_money_flow_zero_cross_relay_support as s
def test_twiggs_uses_prior_close_true_range_and_ema():
 old=s.P["tmf_periods"];s.P["tmf_periods"]=2
 try:
  h=pd.Series([10.,12.,11.]);l=pd.Series([8.,10.,9.]);c=pd.Series([9.,11.,10.]);v=pd.Series([2.,4.,8.]);valid=pd.Series([True]*3);x=s.twiggs_money_flow(h,l,c,v,valid)
  assert np.isnan(x.tmf.iloc[0]) and x.true_high.iloc[1]==12 and x.true_low.iloc[1]==9
  assert np.isclose(x.adjusted_money_flow.iloc[1],4/3)
  assert np.isfinite(x.tmf.iloc[2])
 finally:s.P["tmf_periods"]=old
def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1
def panel():return pd.DataFrame({"source_valid":[True]*7,"tmf":[0.,.1,.2,-.1,.1,.2,.2],"ordinary_tmf":[.1,.1,0.,-.1,.1,.1,-.1],"variation_rank":[.8,.8,.8,.8,.4,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[1] and z.iloc[1]==1 and a.iloc[3] and z.iloc[3]==-1 and not a.iloc[4]
 u,side,_=s.active(panel(),"ordinary_high_low_bounds");assert u.iloc[3] and side.iloc[3]==-1 and not u.iloc[4]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[2] and side.iloc[2]==1
def test_prepare_requires_nonnegative_volume():
 f=pd.DataFrame({"ts":["2024-01-01T00:00:00Z"],"open":[1.],"high":[2.],"low":[.5],"close":[1.5],"volume":[-1.]});assert not s.prepare(f).row_valid.iloc[0]
