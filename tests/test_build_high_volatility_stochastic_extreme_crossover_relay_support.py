import numpy as np
import pandas as pd
from training import build_high_volatility_stochastic_extreme_crossover_relay_support as s

def test_stochastic_range_and_signal():
 old=(s.P["range_periods"],s.P["signal_periods"]);s.P["range_periods"]=3;s.P["signal_periods"]=2
 try:
  h=pd.Series([10.,12.,14.,16.]);l=pd.Series([4.,6.,8.,10.]);c=pd.Series([7.,9.,13.,11.]);v=pd.Series([True]*4);x=s.stochastic(h,l,c,v)
  assert np.isclose(x.percent_k.iloc[2],90) and np.isclose(x.percent_k.iloc[3],50)
  assert np.isclose(x.percent_d.iloc[3],70)
 finally:s.P["range_periods"],s.P["signal_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*6,"difference":[-2.,-1.,1.,2.,-1.,-2.],"percent_k":[50.,10.,15.,50.,85.,50.],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 assert not s.active(panel(),"no_extreme_gate")[0].iloc[3]
 stale,side,_=s.active(panel(),"one_bar_stale_crossover");assert stale.iloc[3] and side.iloc[3]==1
