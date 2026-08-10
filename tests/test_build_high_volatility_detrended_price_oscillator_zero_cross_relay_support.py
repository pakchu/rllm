import numpy as np
import pandas as pd
from training import build_high_volatility_detrended_price_oscillator_zero_cross_relay_support as s

def test_cross_side():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 values=pd.Series([1.,2.,3.]);result=s.causal_sma(values,pd.Series([True]*3),2);assert np.isnan(result.iloc[0]) and result.iloc[2]==2.5

def test_dpo_frozen_formula_and_gap_reset():
 close=pd.Series(np.arange(1.,25.));valid=pd.Series([True]*24);sma=s.causal_sma(close,valid,20);dpo=close.shift(11)-sma
 assert np.isnan(dpo.iloc[18]) and dpo.iloc[19]==9.-10.5
 gap=s.causal_sma(close,pd.Series([True]*10+[False]+[True]*13),20);assert gap.iloc[-1]!=gap.iloc[-1]

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"current_close_sma_side":[0,-1,0,1,0,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 current,side,_=s.active(panel(),"current_close_sma_cross");assert current.iloc[1] and side.iloc[1]==-1 and current.iloc[3]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
