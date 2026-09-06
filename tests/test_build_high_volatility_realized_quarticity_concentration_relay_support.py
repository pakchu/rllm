import hashlib
import numpy as np
import pandas as pd
from training import build_high_volatility_realized_quarticity_concentration_relay_support as s

def test_quarticity_concentration_is_scale_free_and_detects_concentration():
 diffuse=np.ones(96);concentrated=np.r_[np.ones(95),10.]
 assert np.isclose(s.quarticity_concentration(diffuse),1)
 assert np.isclose(s.quarticity_concentration(diffuse*7),1)
 assert s.quarticity_concentration(concentrated)>1

def test_prior_rank_excludes_current(monkeypatch):
 monkeypatch.setitem(s.P,"minimum_history_decisions",2);monkeypatch.setitem(s.P,"variation_history_decisions",3)
 r=s.prior_rank(pd.Series([1.,2.,np.nan,3.]));assert np.isnan(r.iloc[1]) and r.iloc[3]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"entry_side":[0,-1,1,1,-1,1,1],"upper_tail_onset":[False,False,True,False,False,False,False],"diffuse_onset":[False,False,False,False,True,False,False],"variation_rank":[.8,.8,.8,.8,.8,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 diffuse,side,_=s.active(panel(),"diffuse_variation");assert diffuse.iloc[4] and side.iloc[4]==-1
 stale,side,_=s.active(panel(),"one_decision_stale_onset");assert stale.iloc[3] and side.iloc[3]==1
 forced,side,_=s.active(panel(),"forced_long");assert side[forced].eq(1).all()

def test_source_is_blind_and_hash_bound():
 assert "funding" not in s.QUERY.lower() and "gross9" not in s.QUERY.lower()
 assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
