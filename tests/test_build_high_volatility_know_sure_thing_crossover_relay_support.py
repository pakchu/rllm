import numpy as np
import pandas as pd
from training import build_high_volatility_know_sure_thing_crossover_relay_support as s

def test_cross_side():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 old=dict(s.P);s.P.update(roc_periods=[1,2,3,4],sma_periods=[2,2,2,2],weights=[1,2,3,4],signal_periods=2)
 try: values=s.kst_values(pd.Series(np.arange(1,12,dtype=float)),pd.Series([True]*11));assert np.isnan(values.kst.iloc[4]) and values.kst.iloc[-1]>0 and np.isfinite(values.difference.iloc[-1])
 finally: s.P.clear();s.P.update(old)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"zero_line_side":[0,0,1,0,-1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
