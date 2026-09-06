import numpy as np
import pandas as pd
from training import build_high_volatility_zero_lag_exponential_moving_average_price_crossover_relay_support as s

def test_lean_ema_seeds_with_sma_then_recurses():
 ema=s.LeanEma(3);assert np.isnan(ema.update(1)) and np.isnan(ema.update(2));assert ema.update(3)==2;assert ema.update(5)==3.5

def test_zlema_adjusts_by_four_bar_delay_and_uses_frozen_warmup():
 close=pd.Series(np.arange(1.,18.));frame=s.zlema_price_cross(close,pd.Series([True]*len(close)))
 assert frame.loc[4,"adjusted_close"]==9
 assert frame.zlema.first_valid_index()==14
 expected=np.mean([2*x-(x-4) for x in range(5,15)])
 expected=(2/11)*(2*15-11)+(9/11)*expected
 assert np.isclose(frame.loc[14,"zlema"],expected)

def test_invalid_bar_resets_delay_and_ema():
 close=pd.Series(np.arange(1.,33.));valid=pd.Series([True]*16+[False]+[True]*15)
 frame=s.zlema_price_cross(close,valid)
 assert frame.zlema.first_valid_index()==14 and frame.loc[17:].zlema.first_valid_index()==31

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"ordinary_ema_side":[0,-1,0,1,0,-1],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
 follow,side,_=s.active(panel(),"direction_flip");assert follow.iloc[2] and side.iloc[2]==-1
 zero,side,_=s.active(panel(),"ordinary_ema_crossover");assert zero.iloc[1] and side.iloc[1]==-1
