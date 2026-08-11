import numpy as np
import pandas as pd
from training import build_high_volatility_augen_price_spike_fade_relay_support as s

def test_augen_price_spike_uses_three_lagged_log_returns_and_fades_two_sigma():
 closes=pd.Series([100.,101.,103.,106.,120.,121.])
 frame=s.augen_price_spike(closes,pd.Series([True]*len(closes)))
 expected=np.std(np.log(np.array([101/100,103/101,106/103])),ddof=0)
 assert frame.local_volatility.iloc[4]==expected
 assert frame.augen_price_spike.iloc[4]==(120-106)/(expected*106)
 assert frame.entry_side.iloc[4]==-1

def test_augen_price_spike_does_not_bridge_invalid_bar():
 frame=s.augen_price_spike(pd.Series([100.,101.,103.,106.,120.,121.]),pd.Series([True,True,False,True,True,True]))
 assert frame.local_volatility.isna().all()

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"zero_cross_side":[0,-1,0,1,0,-1],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
 follow,side,_=s.active(panel(),"direction_follow");assert follow.iloc[2] and side.iloc[2]==-1
 zero,side,_=s.active(panel(),"zero_cross");assert zero.iloc[1] and side.iloc[1]==-1
