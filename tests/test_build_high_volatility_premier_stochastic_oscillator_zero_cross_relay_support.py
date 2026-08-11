import numpy as np
import pandas as pd
from training import build_high_volatility_premier_stochastic_oscillator_zero_cross_relay_support as s

def test_lean_ema_seeds_with_sma_then_recurses():
 ema=s.LeanEma(3)
 assert np.isnan(ema.update(1)) and np.isnan(ema.update(2))
 assert ema.update(3)==2
 assert ema.update(5)==3.5

def test_premier_stochastic_readiness_and_invalid_reset():
 old=s.P.copy();s.P.update({"stochastic_periods":3,"ema_periods":2})
 try:
  high=pd.Series([2.,3.,4.,5.,6.,7.,8.,9.,10.]);low=high-1;close=high-.25
  frame=s.premier_stochastic(high,low,close,pd.Series([True]*len(high)))
  assert frame.pso.first_valid_index()==4
  broken=s.premier_stochastic(high,low,close,pd.Series([True,True,True,False,True,True,True,True,True]))
  assert broken.pso.first_valid_index()==8
 finally:s.P.clear();s.P.update(old)

def test_default_chain_has_lean_18_bar_warmup():
 high=pd.Series(100+np.arange(30)*.2+np.where(np.arange(30)%2,1.,-1.));low=high-2;close=(high+low)/2
 frame=s.premier_stochastic(high,low,close,pd.Series([True]*len(high)))
 assert frame.pso.first_valid_index()==17

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"raw_fast_k_side":[0,-1,0,1,0,-1],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
 follow,side,_=s.active(panel(),"direction_flip");assert follow.iloc[2] and side.iloc[2]==-1
 zero,side,_=s.active(panel(),"raw_fast_k_midline");assert zero.iloc[1] and side.iloc[1]==-1
