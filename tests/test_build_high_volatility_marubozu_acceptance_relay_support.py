import hashlib
import numpy as np
import pandas as pd
from training import build_high_volatility_marubozu_acceptance_relay_support as s

def test_marubozu_geometry():
 x=s.marubozu(pd.Series([10.,20.,30.]),pd.Series([20.,20.,40.]),pd.Series([10.,10.,30.]),pd.Series([20.,10.,35.]))
 assert x.pattern.tolist()==[True,True,False] and x.body_side.tolist()==[1,-1,1]
 assert x.upper_wick_share.iloc[0]==0 and x.lower_wick_share.iloc[1]==0

def test_prior_rank_excludes_current(monkeypatch):
 monkeypatch.setitem(s.P,"minimum_variation_history_decisions",2);monkeypatch.setitem(s.P,"variation_history_decisions",3)
 r=s.prior_rank(pd.Series([1.,2.,np.nan,3.]));assert np.isnan(r.iloc[1]) and r.iloc[3]==1

def panel():return pd.DataFrame({"source_valid":[True]*6,"pattern":[False,True,False,True,True,False],"body_side":[0,1,0,-1,1,0],"wick_side":[0,-1,0,1,0,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,True,False,True,False,False] and z[a].tolist()==[1,-1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 wick,side,_=s.active(panel(),"wick_pressure_side");assert wick.iloc[1] and wick.iloc[3] and side[wick].tolist()==[-1,1]
 stale,side,_=s.active(panel(),"one_bar_stale_pattern");assert stale.iloc[2] and side.iloc[2]==1
 forced,side,_=s.active(panel(),"forced_long");assert side[forced].eq(1).all()

def test_source_is_blind_and_hash_bound():
 assert "funding" not in s.QUERY.lower() and "gross9" not in s.QUERY.lower()
 assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
