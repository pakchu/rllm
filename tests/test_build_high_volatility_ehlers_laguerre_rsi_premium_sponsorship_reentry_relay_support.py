import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_laguerre_rsi_premium_sponsorship_reentry_relay_support as s

def test_laguerre_recursion_and_bounds():
 x=s.laguerre_rsi(pd.Series([10.,12.,11.]),pd.Series([True]*3))
 assert x.loc[0,["l0","l1","l2","l3"]].tolist()==[10.]*4 and np.isnan(x.laguerre_rsi.iloc[0])
 assert x.loc[1,["l0","l1","l2","l3"]].tolist()==[11.,9.5,10.25,9.875]
 assert x.laguerre_rsi.dropna().between(0,1).all()

def test_laguerre_resets_after_invalid_gap():
 x=s.laguerre_rsi(pd.Series([10.,12.,99.,20.,21.]),pd.Series([True,True,False,True,True]))
 assert x.iloc[2].isna().all() and x.loc[3,["l0","l1","l2","l3"]].tolist()==[20.]*4
 assert np.isnan(x.laguerre_rsi.iloc[3]) and np.isfinite(x.laguerre_rsi.iloc[4])

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"laguerre_rsi":[.1,.15,.3,.5,.9,.75,.5],"variation_rank":[.8,.8,.8,.8,.8,.4,.8],"premium_displacement":[1.,1.,1.,1.,1.,-1.,1.]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[5]
 outward,side,_=s.active(panel(),"outward_crossing");assert outward.iloc[4] and side.iloc[4]==1
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[3] and side.iloc[3]==1
