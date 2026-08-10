import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_deviation_scaled_fisher_reentry_relay_support as s

def test_deviation_scaled_fisher_recursion_and_reset():
 assert s.reentry_side(pd.Series([-3.,-1.,0.,3.,1.]),2.,-2.).tolist()==[0,1,0,0,-1]
 prices=pd.Series(100+np.sin(np.arange(130)/4)+np.arange(130)*.01);x=s.deviation_scaled_fisher(prices,pd.Series([True]*70+[False]+[True]*59))
 assert x.zeros.iloc[:2].isna().all() and np.isfinite(x.zeros.iloc[2]) and np.isfinite(x.scaled.iloc[41])
 assert x.iloc[70][["zeros","fisher"]].isna().all() and x.run_length.iloc[71:74].tolist()==[1,2,3] and np.isfinite(x.scaled.iloc[112])

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"raw_entry_side":[0,-1,0,1,0,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 raw,side,_=s.active(panel(),"unfisherized_equivalent_reentry");assert raw.iloc[1] and side.iloc[1]==-1 and raw.iloc[3] and side.iloc[3]==1
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[3] and side.iloc[3]==1
