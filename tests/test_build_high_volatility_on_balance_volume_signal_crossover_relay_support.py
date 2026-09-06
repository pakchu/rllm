import numpy as np
import pandas as pd
from training import build_high_volatility_on_balance_volume_signal_crossover_relay_support as s

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"difference":[-2.,-1.,1.,2.,-1.,-2.],"signed_volume_side":[0,0,1,0,-1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1

def test_causal_obv_resets_on_gap():
 result=s.causal_obv(pd.Series([10.,11.,12.,9.]),pd.Series([1.,2.,3.,4.]),pd.Series([True,True,False,True]));assert result.obv.tolist()[:2]==[0.,2.] and np.isnan(result.obv.iloc[2]) and result.obv.iloc[3]==0
