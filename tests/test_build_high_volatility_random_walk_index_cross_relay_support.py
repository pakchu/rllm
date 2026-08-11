import hashlib,json,math
import numpy as np
import pandas as pd
from training import build_high_volatility_random_walk_index_cross_relay_support as s


def test_canonical_rwi_uses_prior_atr_and_all_horizons(monkeypatch):
    monkeypatch.setitem(s.P,'maximum_periods',2);monkeypatch.setitem(s.P,'minimum_periods',1)
    bars=pd.DataFrame({'bar_high':[11.,12.,13.,15.],'bar_low':[9.,10.,11.,12.],'bar_close':[10.,11.,12.,14.],'valid_bar':[True]*4})
    r=s.random_walk_index(bars)
    # Prior TR values at rows 1 and 2 are both 2. Current row 3:
    # high candidates: n1=(15-11)/2=2; n2=(15-10)/(2*sqrt(2)).
    assert np.isnan(r.rwi_high.iloc[2])
    assert np.isclose(r.rwi_high.iloc[3],2.)
    assert np.isclose(r.rwi_low.iloc[3],.5)
    assert np.isclose(r.rwi14_high.iloc[3],5/(2*math.sqrt(2)))


def test_rwi_gap_resets(monkeypatch):
    monkeypatch.setitem(s.P,'maximum_periods',2);monkeypatch.setitem(s.P,'minimum_periods',1)
    bars=pd.DataFrame({'bar_high':[11.,12.,13.,14.,15.,16.,17.],'bar_low':[9.,10.,11.,12.,13.,14.,15.],'bar_close':[10.,11.,12.,13.,14.,15.,16.],'valid_bar':[True,True,True,True,False,True,True]})
    r=s.random_walk_index(bars);assert np.isfinite(r.rwi_high.iloc[3]);assert r.rwi_high.iloc[4:].isna().all()


def test_prior_rank_excludes_current_and_resets(monkeypatch):
    monkeypatch.setitem(s.P,'minimum_variation_history_decisions',2);monkeypatch.setitem(s.P,'variation_history_decisions',3)
    r=s.prior_rank(pd.Series([1.,2.,3.,np.nan,4.,5.,6.]),pd.Series([True,True,True,False,True,True,True]));assert np.isnan(r.iloc[1]) and r.iloc[2]==1 and np.isnan(r.iloc[5]) and r.iloc[6]==1


def test_cross_and_frozen_controls():
    assert s.strict_flip_side(pd.Series([-1.,1.,2.,np.nan,-1.,1.]),pd.Series([True,True,True,False,True,True])).tolist()==[0,1,0,0,0,1]
    p=pd.DataFrame({'source_valid':[True]*4,'variation_rank':[.8,.8,.4,.8],'entry_side':[-1,1,-1,1],'fixed_14_side':[1,0,-1,0],'cross':[True]*4})
    a,z,_=s.active(p);assert a.tolist()==[True,True,False,True] and z[a].tolist()==[-1,1,1]
    assert s.active(p,'no_variation_gate')[0].all()
    a,z,_=s.active(p,'fixed_14_only');assert a.tolist()==[True,False,False,False] and z[a].tolist()==[1]
    a,z,_=s.active(p,'one_bar_stale_cross');assert a.tolist()==[False,True,True,False] and z[a].tolist()==[-1,1]
    a,z,_=s.active(p,'direction_flip');assert z[a].tolist()==[1,-1,-1]


def test_source_blind_and_hash_bound():
    q=s.QUERY.lower();assert 'open,high,low,close' in q and 'funding' not in q and 'gross9' not in q
    assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
    v={'한글':'RWI'};expected=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert s.canonical_hash(v)==expected
