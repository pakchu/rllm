import hashlib,json,math
import numpy as np
import pandas as pd
from training import build_high_volatility_spot_perpetual_permutation_entropy_leadership_relay_support as s


def test_permutation_entropy_monotone_and_pattern_mix():
 monotone=np.arange(1.,49.);assert s.permutation_entropy(monotone)==0
 alternating=np.resize(np.array([1.,3.,2.]),48);value=s.permutation_entropy(alternating);assert 0<value<=1


def test_ties_resolve_by_earlier_temporal_index():
 tied=np.ones(48);assert s.permutation_entropy(tied)==0
 invalid=np.ones(47);assert math.isnan(s.permutation_entropy(invalid))


def test_prior_rank_excludes_current_and_skips_invalid(monkeypatch):
 monkeypatch.setitem(s.P,'minimum_entropy_history_decisions',2);monkeypatch.setitem(s.P,'entropy_history_decisions',3)
 r=s.prior_rank(pd.Series([1.,2.,np.nan,3.,4.]),pd.Series([True,True,False,True,True]));assert np.isnan(r.iloc[1]) and r.iloc[3]==1 and r.iloc[4]==1


def test_onset_and_frozen_controls():
 assert s.fresh_onset(pd.Series([True,False,True,True]),pd.Series([True]*4)).tolist()==[False,False,True,False]
 p=pd.DataFrame({'source_valid':[True]*5,'variation_tail':[True,True,False,True,True],'entropy_tail':[False,True,True,False,True],'entropy_leadership':[.1,.2,-.3,-.2,.4],'entropy_leadership_rank':[.7,.9,.1,.1,.9],'direction_agreement':[True]*5,'eligible':[False,True,False,False,True],'entry_side':[-1,1,-1,1,-1]})
 a,z,_=s.active(p);assert a.tolist()==[False,True,False,False,True] and z[a].tolist()==[1,-1]
 a,z,_=s.active(p,'no_entropy_leadership_gate');assert a.tolist()==[False,False,False,True,False]
 a,z,_=s.active(p,'no_variation_gate');assert a.tolist()==[False,True,False,False,True]
 a,z,_=s.active(p,'perpetual_more_ordered');assert a.tolist()==[False,False,False,True,False]
 a,z,_=s.active(p,'one_bar_stale_onset');assert a.tolist()==[False,False,True,False,False] and z[a].tolist()==[1]
 a,z,_=s.active(p,'direction_flip');assert z[a].tolist()==[-1,1]
 a,z,_=s.active(p,'forced_long');assert z[a].eq(1).all()


def test_source_blind_and_hash_bound():
 assert 'bars_binance_spot' in s.SPOT_QUERY and 'bars_binance ' in s.PERP_QUERY and 'funding' not in (s.SPOT_QUERY+s.PERP_QUERY).lower() and 'gross9' not in (s.SPOT_QUERY+s.PERP_QUERY).lower()
 assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest();value={'한글':'PE'};expected=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert s.canonical_hash(value)==expected
