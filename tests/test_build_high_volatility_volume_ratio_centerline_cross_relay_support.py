import hashlib
import numpy as np
import pandas as pd
from training import build_high_volatility_volume_ratio_centerline_cross_relay_support as s

def test_volume_ratio_formula(monkeypatch):
 monkeypatch.setitem(s.P,"volume_ratio_periods",3)
 close=pd.Series([10.,11.,10.,10.]);volume=pd.Series([1.,2.,3.,4.]);x=s.volume_ratio_values(close,volume,pd.Series([True]*4))
 assert x.up_sum.iloc[3]==2 and x.down_sum.iloc[3]==3 and x.unchanged_sum.iloc[3]==4
 assert np.isclose(x.volume_ratio.iloc[3],100*(2+2)/(3+2)) and x.net_volume.iloc[3]==-1

def test_zero_cross_side():
 assert s.zero_cross_side(pd.Series([-1.,0.,1.,2.,0.,-1.])).tolist()==[0,0,1,0,0,-1]

def test_prior_rank_excludes_current(monkeypatch):
 monkeypatch.setitem(s.P,"minimum_variation_history_decisions",2);monkeypatch.setitem(s.P,"variation_history_decisions",3)
 r=s.prior_rank(pd.Series([1.,2.,np.nan,3.]));assert np.isnan(r.iloc[1]) and r.iloc[3]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"ratio_side":[0,0,1,0,-1,0,0],"net_side":[0,1,0,0,0,-1,0],"variation_rank":[.8,.8,.8,.8,.4,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 net,side,_=s.active(panel(),"net_volume_zero_cross");assert net.iloc[1] and net.iloc[5] and side[net].tolist()==[1,-1]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
 forced,side,_=s.active(panel(),"forced_long");assert side[forced].eq(1).all()

def test_source_is_blind_and_hash_bound():
 assert "funding" not in s.QUERY.lower() and "gross9" not in s.QUERY.lower()
 assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
