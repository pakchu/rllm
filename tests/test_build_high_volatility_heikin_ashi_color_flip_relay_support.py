import numpy as np
import pandas as pd
from training import build_high_volatility_heikin_ashi_color_flip_relay_support as s

def test_heikin_ashi_recursion_and_reset():
 frame=s.heikin_ashi(pd.Series([1.,2.,3.]),pd.Series([2.,3.,4.]),pd.Series([.5,1.5,2.5]),pd.Series([1.5,2.5,3.5]),pd.Series([True,True,False]))
 assert np.isclose(frame.ha_open.iloc[0],1.25) and np.isclose(frame.ha_open.iloc[1],(1.25+1.25)/2) and np.isnan(frame.ha_open.iloc[2])

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"color":[-1,-1,1,1,-1,-1],"raw_color":[-1,-1,1,1,-1,-1],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_flip");assert stale.iloc[3] and side.iloc[3]==1
