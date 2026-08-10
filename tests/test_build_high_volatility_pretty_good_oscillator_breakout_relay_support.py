import numpy as np
import pandas as pd
from training import build_high_volatility_pretty_good_oscillator_breakout_relay_support as s

def test_cross_side():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 old=dict(s.P);s.P.update(periods=2,breakout_threshold=3.)
 bars=pd.DataFrame({"bar_high":[2.,3.,5.,9.],"bar_low":[1.,2.,3.,4.],"bar_close":[1.5,2.5,4.,8.]})
 try: values=s.pgo_values(bars,pd.Series([True]*4));assert np.isnan(values.pgo.iloc[1]) and np.isfinite(values.pgo.iloc[-1]);assert s.breakout_side(pd.Series([0.,3.1,2.,-3.1])).tolist()==[0,1,0,-1]
 finally:s.P.clear();s.P.update(old)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"zero_line_side":[0,0,1,0,-1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_breakout");assert stale.iloc[3] and side.iloc[3]==1
