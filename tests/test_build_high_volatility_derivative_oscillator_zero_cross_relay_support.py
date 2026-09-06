import numpy as np
import pandas as pd
from training import build_high_volatility_derivative_oscillator_zero_cross_relay_support as s

def test_lean_average_seeds_with_sma_then_recurses():
 ema=s.LeanAverage(3,.5)
 assert np.isnan(ema.update(1)) and np.isnan(ema.update(2))
 assert ema.update(3)==2
 assert ema.update(5)==3.5

def test_derivative_oscillator_warmup_and_invalid_reset():
 old=s.P.copy();s.P.update({"rsi_periods":3,"first_ema_periods":2,"second_ema_periods":2,"signal_periods":2})
 try:
  close=pd.Series([100.,102.,101.,104.,102.,105.,103.,106.,104.,107.,105.,108.,106.,109.,107.,110.])
  frame=s.derivative_oscillator(close,pd.Series([True]*len(close)))
  assert frame.derivative_oscillator.first_valid_index()==6
  broken=s.derivative_oscillator(close,pd.Series([True]*8+[False]+[True]*7))
  assert broken.derivative_oscillator.first_valid_index()==6
  assert broken.derivative_oscillator.iloc[9:15].isna().all()
  assert np.isfinite(broken.derivative_oscillator.iloc[15])
 finally:s.P.clear();s.P.update(old)

def test_default_chain_has_lean_29_bar_warmup():
 close=pd.Series(100+np.arange(40)*.2+np.where(np.arange(40)%2,1.,-1.))
 frame=s.derivative_oscillator(close,pd.Series([True]*len(close)))
 assert frame.derivative_oscillator.first_valid_index()==28

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"raw_rsi_side":[0,-1,0,1,0,-1],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
 follow,side,_=s.active(panel(),"direction_flip");assert follow.iloc[2] and side.iloc[2]==-1
 zero,side,_=s.active(panel(),"raw_rsi_midline");assert zero.iloc[1] and side.iloc[1]==-1
