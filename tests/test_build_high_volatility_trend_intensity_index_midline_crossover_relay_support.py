import numpy as np
import pandas as pd
from training import build_high_volatility_trend_intensity_index_midline_crossover_relay_support as s

def test_trend_intensity_definition():
 old=(s.P["major_periods"],s.P["minor_periods"]);s.P["major_periods"]=2;s.P["minor_periods"]=2
 try:
  x=s.trend_intensity(pd.Series([1.,3.,1.,3.]),pd.Series([True]*4))
  assert np.isnan(x.tii.iloc[1]) and np.isclose(x.tii.iloc[2],50.) and np.isclose(x.tii.iloc[3],50.)
 finally:s.P["major_periods"],s.P["minor_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*6,"tii":[40.,45.,55.,60.,45.,40.],"close_sma_difference":[-2.,-1.,1.,2.,-1.,-2.],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[2] and z.iloc[2]==1 and not a.iloc[4]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_crossover");assert stale.iloc[3] and side.iloc[3]==1
