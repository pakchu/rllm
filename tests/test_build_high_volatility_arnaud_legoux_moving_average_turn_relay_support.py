import numpy as np
import pandas as pd
from training import build_high_volatility_arnaud_legoux_moving_average_turn_relay_support as s

def test_alma_average_uses_offset_gaussian_weights():
 result=s.alma_average(pd.Series(np.arange(1.,10.)),pd.Series([True]*9),9,.85,6.)
 assert result.iloc[-1]>5 and result.iloc[-1]<9

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"slope":[-2.,-1.,1.,2.,-1.,-2.],"sma_9_slope":[-2.,-1.,1.,2.,-1.,-2.],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_turn");assert stale.iloc[3] and side.iloc[3]==1
