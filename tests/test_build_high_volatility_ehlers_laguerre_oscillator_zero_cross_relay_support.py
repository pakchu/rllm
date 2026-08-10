import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_laguerre_oscillator_zero_cross_relay_support as s

def test_laguerre_filter_seed_recursion_and_reset():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 prices=pd.Series(100+np.sin(np.arange(220)/4)+np.arange(220)*.01);x=s.laguerre_filter(prices,pd.Series([True]*110+[False]+[True]*109),.5,30)
 assert x.ultimate_smoother.iloc[:3].tolist()==prices.iloc[:3].tolist() and np.isfinite(x.laguerre_oscillator.iloc[99])
 assert x.iloc[110][["ultimate_smoother","laguerre_oscillator"]].isna().all() and x.run_length.iloc[111:114].tolist()==[1,2,3]
 assert x.laguerre_oscillator.iloc[209]!=x.laguerre_oscillator.iloc[209] and np.isfinite(x.laguerre_oscillator.iloc[210])

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"alternate_entry_side":[0,-1,0,1,0,0],"rate_entry_side":[0,0,-1,0,1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 raw,side,_=s.active(panel(),"alternate_0_8_20");assert raw.iloc[1] and side.iloc[1]==-1 and raw.iloc[3] and side.iloc[3]==1
 roc,side,_=s.active(panel(),"oscillator_roc_turn");assert roc.iloc[2] and side.iloc[2]==-1
