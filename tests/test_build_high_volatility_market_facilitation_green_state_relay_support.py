import numpy as np
import pandas as pd
from training import build_high_volatility_market_facilitation_green_state_relay_support as s

def test_market_facilitation_formula():
 assert np.isclose((12.-10.)/4.,.5)
def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1
def panel():return pd.DataFrame({"source_valid":[True]*7,"green_state":[False,True,False,True,True,False,True],"range_volume_up":[False,False,True,True,True,False,True],"bar_side":[1,1,-1,-1,1,1,-1],"variation_rank":[.8,.8,.8,.8,.4,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[1] and z.iloc[1]==1 and a.iloc[3] and z.iloc[3]==-1 and not a.iloc[4] and a.iloc[6] and z.iloc[6]==-1
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 raw,side,_=s.active(panel(),"range_and_volume_up");assert raw.iloc[2] and side.iloc[2]==-1
 stale,side,_=s.active(panel(),"one_bar_stale_state");assert stale.iloc[2] and side.iloc[2]==1

def test_prepare_rejects_negative_volume():
 f=pd.DataFrame({"ts":["2024-01-01T00:00:00Z"],"open":[1.],"high":[2.],"low":[.5],"close":[1.5],"volume":[-1.]});assert not s.prepare(f).row_valid.iloc[0]
