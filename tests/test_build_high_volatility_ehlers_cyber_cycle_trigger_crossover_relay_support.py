import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_cyber_cycle_trigger_crossover_relay_support as s

def test_cyber_cycle_uses_published_initialization_and_trigger():
 old=s.P.copy();s.P.update({"alpha":.07})
 try:
  frame=s.cyber_cycle(pd.Series([1.,2.,4.,7.,11.,16.,22.,29.]),pd.Series([True]*8))
  assert np.isnan(frame.cycle.iloc[2])
  assert np.isfinite(frame.cycle.iloc[6])
  assert np.isclose(frame.trigger.iloc[7],frame.cycle.iloc[6])
 finally:s.P.clear();s.P.update(old)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"difference":[-2.,-1.,1.,2.,-1.,-2.],"cycle":[-2.,-1.,1.,2.,-1.,-2.],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
