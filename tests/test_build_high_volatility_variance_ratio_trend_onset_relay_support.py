import hashlib
import numpy as np
import pandas as pd
from training import build_high_volatility_variance_ratio_trend_onset_relay_support as s

def test_variance_ratio_identity_and_persistence():
 alternating=np.array([1.,-1.]*20);persistent=np.repeat([1.,-1.],20)
 vr_alt,one_alt,_=s.variance_ratio(alternating,2);vr_p,one_p,_=s.variance_ratio(persistent,2)
 assert one_alt>0 and one_p>0 and vr_alt<1 and vr_p>1

def test_prior_rank_excludes_current(monkeypatch):
 monkeypatch.setitem(s.P,"minimum_variation_history_decisions",2);monkeypatch.setitem(s.P,"variation_history_decisions",3)
 r=s.prior_rank(pd.Series([1.,2.,np.nan,3.]));assert np.isnan(r.iloc[1]) and r.iloc[3]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"entry_side":[0,-1,1,1,-1,1,1],"above_unity":[False,False,True,False,False,False,False],"below_unity":[False,False,False,False,True,False,False],"variation_rank":[.8,.8,.8,.8,.8,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 below,side,_=s.active(panel(),"below_unity_reversion");assert below.iloc[4] and side.iloc[4]==1
 stale,side,_=s.active(panel(),"one_bar_stale_onset");assert stale.iloc[3] and side.iloc[3]==1
 forced,side,_=s.active(panel(),"forced_long");assert side[forced].eq(1).all()

def test_source_is_blind_and_hash_bound():
 assert "funding" not in s.QUERY.lower() and "gross9" not in s.QUERY.lower()
 assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
