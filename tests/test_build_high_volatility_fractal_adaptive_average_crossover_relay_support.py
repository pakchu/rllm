import numpy as np
import pandas as pd
from training import build_high_volatility_fractal_adaptive_average_crossover_relay_support as s

def test_fractal_average_is_causal_and_bounded():
 old=s.P.copy();s.P.update({"frama_periods":4,"half_periods":2,"dimension_exponent":4.6,"alpha_min":.01,"alpha_max":1.0})
 try:
  frame=s.fractal_average(pd.Series([2.,3.,4.,5.,6.]),pd.Series([1.,2.,3.,4.,5.]),pd.Series([1.5,2.5,3.5,4.5,5.5]),pd.Series([True]*5));assert np.isfinite(frame.frama.iloc[3]) and .01<=frame.alpha.iloc[3]<=1
 finally:s.P.clear();s.P.update(old)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"difference":[-2.,-1.,1.,2.,-1.,-2.],"fixed_difference":[-2.,-1.,1.,2.,-1.,-2.],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_crossover");assert stale.iloc[3] and side.iloc[3]==1
