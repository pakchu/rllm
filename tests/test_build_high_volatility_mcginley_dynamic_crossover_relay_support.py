import numpy as np
import pandas as pd
from training import build_high_volatility_mcginley_dynamic_crossover_relay_support as s

def test_mcginley_uses_frozen_nonlinear_step():
 old=s.P.copy();s.P.update({"dynamic_periods":2,"adjustment_constant":.5,"ratio_power":4})
 try:
  result=s.mcginley(pd.Series([10.,11.]),pd.Series([True,True]));expected=10+(1)/(1*(1.1**4));assert np.isclose(result.iloc[1],expected)
 finally:s.P.clear();s.P.update(old)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"difference":[-2.,-1.,1.,2.,-1.,-2.],"simple_difference":[-2.,-1.,1.,2.,-1.,-2.],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_crossover");assert stale.iloc[3] and side.iloc[3]==1
