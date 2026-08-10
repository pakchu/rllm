import numpy as np
import pandas as pd
from training import build_high_volatility_ultimate_oscillator_reentry_relay_support as s

def test_ultimate_oscillator_uses_canonical_weighting():
 old=(s.P["short_periods"],s.P["medium_periods"],s.P["long_periods"]);s.P["short_periods"]=s.P["medium_periods"]=s.P["long_periods"]=1
 try:
  h=pd.Series([10.,12.]);l=pd.Series([8.,9.]);c=pd.Series([9.,11.]);v=pd.Series([True,True]);x=s.ultimate_oscillator(h,l,c,v);assert x.buying_pressure.iloc[1]==2 and x.true_range.iloc[1]==3 and np.isclose(x.ultimate_oscillator.iloc[1],200/3) and np.isclose(x.equal_weight_oscillator.iloc[1],200/3)
 finally:s.P["short_periods"],s.P["medium_periods"],s.P["long_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"ultimate_oscillator":[30.,35.,50.,70.,65.,30.,35.],"equal_weight_oscillator":[50.,50.,30.,35.,50.,70.,65.],"variation_rank":[.8,.8,.8,.8,.4,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[1] and z.iloc[1]==1 and not a.iloc[4] and a.iloc[6] and z.iloc[6]==1
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 equal,side,_=s.active(panel(),"equal_horizon_weights");assert equal.iloc[3] and side.iloc[3]==1 and equal.iloc[6] and side.iloc[6]==-1
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[2] and side.iloc[2]==1
