import numpy as np
import pandas as pd
from training import build_high_volatility_money_flow_index_reentry_relay_support as s

def test_money_flow_index_weights_quote_volume_and_maps_zero_denominators():
 old=s.P["mfi_periods"]; s.P["mfi_periods"]=2
 try:
  h=pd.Series([10.,11.,10.,12.]); l=h-2; c=h-1; v=pd.Series([1.,10.,1.,1.]); valid=pd.Series([True]*4); x=s.money_flow_index(h,l,c,v,valid)
  assert x.positive_flow.iloc[1]==100 and x.negative_flow.iloc[2]==9
  assert np.isclose(x.mfi.iloc[2],100-100/(1+100/9))
  assert x.mfi.iloc[3]==55
  rising=s.money_flow_index(pd.Series([10.,11.,12.]),pd.Series([8.,9.,10.]),pd.Series([9.,10.,11.]),pd.Series([1.,1.,1.]),pd.Series([True]*3)); assert rising.mfi.iloc[2]==100
 finally:s.P["mfi_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float))); assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel(): return pd.DataFrame({"source_valid":[True]*8,"mfi":[20.,25.,50.,80.,75.,50.,20.,25.],"unweighted_mfi":[50.,50.,20.,25.,50.,80.,75.,50.],"variation_rank":[.8,.8,.8,.8,.4,.8,.8,.8]})
def test_controls():
 a,z,_=s.active(panel()); assert a.tolist()==[False,True,False,False,False,False,False,True] and z[a].tolist()==[1,1]
 unweighted,side,_=s.active(panel(),"unweighted_typical_direction"); assert unweighted.iloc[3] and side.iloc[3]==1 and unweighted.iloc[6] and side.iloc[6]==-1
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_reentry"); assert stale.iloc[2] and side.iloc[2]==1
 flipped,side,_=s.active(panel(),"direction_flip"); assert flipped.iloc[1] and side.iloc[1]==-1

def test_prepare_requires_nonnegative_quote_volume():
 frame=pd.DataFrame({"ts":["2024-01-01T00:00:00Z"],"open":[1.],"high":[2.],"low":[.5],"close":[1.5],"quote_asset_volume":[-1.]}); assert not s.prepare(frame).row_valid.iloc[0]
