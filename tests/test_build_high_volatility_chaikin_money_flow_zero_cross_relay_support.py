import numpy as np
import pandas as pd
from training import build_high_volatility_chaikin_money_flow_zero_cross_relay_support as s

def test_cmf_weights_quote_volume_and_handles_flat_bar():
 old=s.P["cmf_periods"];s.P["cmf_periods"]=2
 try:
  h=pd.Series([10.,10.,10.]);l=pd.Series([0.,0.,10.]);c=pd.Series([10.,0.,10.]);v=pd.Series([9.,1.,5.]);valid=pd.Series([True]*3);x=s.chaikin_money_flow(h,l,c,v,valid)
  assert x.money_flow_multiplier.tolist()==[1.,-1.,0.]
  assert x.cmf.iloc[1]==.8 and x.unweighted_cmf.iloc[1]==0
  assert x.cmf.iloc[2]==-1/6
 finally:s.P["cmf_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"cmf":[0.,.1,.2,-.1,.1,.2,.2],"unweighted_cmf":[.1,.1,0.,-.1,.1,.1,-.1],"variation_rank":[.8,.8,.8,.8,.4,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[1] and z.iloc[1]==1 and a.iloc[3] and z.iloc[3]==-1 and not a.iloc[4]
 u,side,_=s.active(panel(),"unweighted_close_location");assert u.iloc[3] and side.iloc[3]==-1 and not u.iloc[4]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[2] and side.iloc[2]==1

def test_prepare_requires_nonnegative_quote_volume():
 f=pd.DataFrame({"ts":["2024-01-01T00:00:00Z"],"open":[1.],"high":[2.],"low":[.5],"close":[1.5],"quote_asset_volume":[-1.]});assert not s.prepare(f).row_valid.iloc[0]
