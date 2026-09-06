import numpy as np
import pandas as pd
from training import build_high_volatility_kaufman_adaptive_average_crossover_relay_support as s

def test_adaptive_average_uses_canonical_efficiency():
 old=s.P.copy();s.P.update({"efficiency_periods":2,"fast_periods":2,"slow_periods":4})
 try:
  frame=s.adaptive_average(pd.Series([1.,2.,3.,2.]),pd.Series([True]*4))
  assert np.isclose(frame.efficiency_ratio.iloc[2],1.0) and np.isclose(frame.efficiency_ratio.iloc[3],0.0)
  assert np.isclose(frame.kama.iloc[2],3.0)
 finally:s.P.clear();s.P.update(old)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"difference":[-2.,-1.,1.,2.,-1.,-2.],"slow_difference":[-2.,-1.,1.,2.,-1.,-2.],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_crossover");assert stale.iloc[3] and side.iloc[3]==1
