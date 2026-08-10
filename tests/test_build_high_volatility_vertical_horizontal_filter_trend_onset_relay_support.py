import numpy as np
import pandas as pd
from training import build_high_volatility_vertical_horizontal_filter_trend_onset_relay_support as s

def test_vhf_formula_and_gap_reset():
 old=s.P["vhf_periods"];s.P["vhf_periods"]=3
 try:
  values=pd.Series([1.,3.,2.,4.,9.,10.,12.]);x=s.vhf_frame(values,pd.Series([True]*4+[False]+[True]*2));assert np.isnan(x.vhf.iloc[2]) and x.close_range.iloc[3]==2 and x.absolute_travel.iloc[3]==5 and x.vhf.iloc[3]==.4 and x.net_displacement.iloc[3]==3
  assert x.iloc[4:].vhf.isna().all()
 finally:s.P["vhf_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"direction_side":[1,-1,1,1,-1,1],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 direction,side,_=s.active(panel(),"direction_only");assert direction.iloc[0] and side.iloc[1]==-1
 stale,side,_=s.active(panel(),"one_bar_stale_onset");assert stale.iloc[3] and side.iloc[3]==1
