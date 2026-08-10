import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_decycler_oscillator_zero_cross_relay_support as s

def test_decycler_filter_seed_recursion_and_reset():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 prices=pd.Series([10.,11.,13.,12.,14.,15.,99.,20.,21.,23.]);x=s.decycler_filter(prices,pd.Series([True]*6+[False]+[True]*3),125,1.)
 assert x.high_pass.iloc[:2].tolist()==[0.,0.] and x.decycler_oscillator.iloc[:2].tolist()==[0.,0.]
 assert np.isfinite(x.high_pass.iloc[2]) and np.isfinite(x.decycler_oscillator.iloc[2])
 assert x.iloc[6][["high_pass","decycler_oscillator"]].isna().all() and x.run_length.iloc[7:9].tolist()==[1,2]
 assert x.high_pass.iloc[7:9].tolist()==[0.,0.] and np.isfinite(x.decycler_oscillator.iloc[9])

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"fast_entry_side":[0,-1,0,1,0,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 raw,side,_=s.active(panel(),"fast_100_decycler");assert raw.iloc[1] and side.iloc[1]==-1 and raw.iloc[3] and side.iloc[3]==1
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
