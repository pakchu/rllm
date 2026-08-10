import numpy as np
import pandas as pd
from training import build_high_volatility_ichimoku_cloud_breakout_relay_support as s

def test_visible_cloud_is_exactly_displaced_source():
 old=(s.P["tenkan_periods"],s.P["kijun_periods"],s.P["span_b_periods"],s.P["cloud_displacement_periods"]);s.P["tenkan_periods"]=2;s.P["kijun_periods"]=3;s.P["span_b_periods"]=4;s.P["cloud_displacement_periods"]=2
 try:
  h=pd.Series(np.arange(10.,18.));l=h-4;v=pd.Series([True]*8);x=s.ichimoku(h,l,v)
  assert x.visible_span_a.iloc[6]==x.source_span_a.iloc[4] and x.visible_span_b.iloc[6]==x.source_span_b.iloc[4]
 finally:s.P["tenkan_periods"],s.P["kijun_periods"],s.P["span_b_periods"],s.P["cloud_displacement_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*6,"bar_close":[9.,9.,11.,12.,8.,7.],"cloud_top":[10.]*6,"cloud_bottom":[8.5]*6,"source_span_a":[10.]*6,"source_span_b":[8.5]*6,"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_breakout");assert stale.iloc[3] and side.iloc[3]==1
