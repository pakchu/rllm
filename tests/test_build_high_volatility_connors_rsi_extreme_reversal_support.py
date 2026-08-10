import numpy as np
import pandas as pd
from training import build_high_volatility_connors_rsi_extreme_reversal_support as s

def test_wilder_rsi_and_rank_are_causal():
 rsi=s.wilder_rsi(pd.Series([1.,2.,3.,2.]),pd.Series([True]*4),2);assert rsi.iloc[2]==100 and 0<rsi.iloc[3]<100
 old=s.P["return_rank_periods"];s.P["return_rank_periods"]=2
 try:
  returns,ranks=s.return_percent_rank(pd.Series([1.,2.,3.,6.]),pd.Series([True]*4));assert np.isnan(ranks.iloc[2]) and ranks.iloc[3]==50
 finally:s.P["return_rank_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"price_rsi_side":[0,0,1,0,-1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_entry");assert stale.iloc[3] and side.iloc[3]==1
