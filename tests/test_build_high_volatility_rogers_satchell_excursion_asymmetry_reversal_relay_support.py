import hashlib,json
import numpy as np
import pandas as pd
from training import build_high_volatility_rogers_satchell_excursion_asymmetry_reversal_relay_support as s

def test_canonical_rogers_satchell_components_and_window(monkeypatch):
 monkeypatch.setitem(s.P,"bars",2)
 bars=pd.DataFrame({"bar_open":[100.,100.,100.],"bar_high":[110.,108.,107.],"bar_low":[90.,92.,93.],"bar_close":[105.,95.,104.],"valid_bar":[True,True,True]})
 x=s.rogers_satchell_states(bars);u=np.log(110/105)*np.log(110/100);l=np.log(90/105)*np.log(90/100)
 assert np.isclose(x.upper_component.iloc[0],u) and np.isclose(x.lower_component.iloc[0],l)
 assert np.isnan(x.rs_variation.iloc[0]) and np.isfinite(x.rs_variation.iloc[1])
 assert np.isclose(x.excursion_asymmetry.iloc[1],(x.upper_energy.iloc[1]-x.lower_energy.iloc[1])/x.total_energy.iloc[1])

def test_invalid_bar_resets_six_bar_window(monkeypatch):
 monkeypatch.setitem(s.P,"bars",2);bars=pd.DataFrame({"bar_open":[100.]*4,"bar_high":[110.]*4,"bar_low":[90.]*4,"bar_close":[105.]*4,"valid_bar":[True,False,True,True]});x=s.rogers_satchell_states(bars)
 assert np.isnan(x.rs_variation.iloc[2]) and np.isfinite(x.rs_variation.iloc[3])

def test_prior_rank_excludes_current_and_resets(monkeypatch):
 monkeypatch.setitem(s.P,"minimum_variation_history_decisions",2);monkeypatch.setitem(s.P,"variation_history_decisions",3)
 r=s.prior_rank(pd.Series([1.,2.,3.,np.nan,4.,5.,6.]),pd.Series([True,True,True,False,True,True,True]));assert np.isnan(r.iloc[1]) and r.iloc[2]==1 and np.isnan(r.iloc[4]) and np.isnan(r.iloc[5]) and r.iloc[6]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*4,"variation_rank":[.8,.8,.4,.8],"asymmetry_rank":[.8,.4,.8,.8],"entry_side":[-1,1,-1,1],"latest_bar_asymmetry":[-.2,.3,-.4,.5],"onset":[True,False,False,True]})

def test_primary_and_frozen_controls():
 p=panel();a,z,_=s.active(p);assert a.tolist()==[True,False,False,True] and z[a].tolist()==[-1,1]
 assert s.active(p,"no_asymmetry_tail")[0].tolist()==[True,True,False,True]
 assert s.active(p,"no_variation_gate")[0].tolist()==[True,False,True,True]
 stale,side,_=s.active(p,"one_bar_stale_onset");assert stale.tolist()==[False,True,False,False] and side[stale].tolist()==[-1]
 follow,side,_=s.active(p,"direction_follow");assert follow.equals(a) and side[follow].tolist()==[1,-1]
 latest,side,_=s.active(p,"latest_bar_asymmetry");assert latest.equals(a) and side[latest].tolist()==[1,-1]
 forced,side,_=s.active(p,"forced_long");assert forced.equals(a) and side[forced].eq(1).all()

def test_source_blind_and_hash_bound():
 q=s.QUERY.lower();assert "open,high,low,close" in q and "funding" not in q and "gross9" not in q
 assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest();v={"한글":"RS"};expected=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert s.canonical_hash(v)==expected
