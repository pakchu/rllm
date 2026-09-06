import hashlib,json,math
import numpy as np
import pandas as pd
from training import build_high_volatility_korean_cash_lead_lag_relay_support as s

def test_pearson_and_invalid():
    x=np.arange(47.);assert s.pearson(x,x)==1 and s.pearson(x,-x)==-1
    assert math.isnan(s.pearson(np.ones(47),np.ones(47))) and math.isnan(s.pearson(np.ones(46),np.ones(47)))

def test_prior_rank_excludes_current_and_skips_invalid(monkeypatch):
    monkeypatch.setitem(s.P,'minimum_leadership_history_decisions',2);monkeypatch.setitem(s.P,'leadership_history_decisions',3)
    r=s.prior_rank(pd.Series([1.,2.,np.nan,3.,4.]),pd.Series([True,True,False,True,True]));assert np.isnan(r.iloc[1]) and r.iloc[3]==1 and r.iloc[4]==1

def test_onset_and_frozen_controls():
    assert s.fresh_onset(pd.Series([True,False,True,True]),pd.Series([True]*4)).tolist()==[False,False,True,False]
    p=pd.DataFrame({'source_valid':[True]*5,'variation_tail':[True,True,False,True,True],'leadership_tail':[False,True,True,False,True],'leadership_advantage':[.1,.2,-.3,-.2,.4],'leadership_rank':[.7,.9,.1,.1,.9],'direction_agreement':[True]*5,'eligible':[False,True,False,False,True],'entry_side':[-1,1,-1,1,-1]})
    a,z,_=s.active(p);assert a.tolist()==[False,True,False,False,True] and z[a].tolist()==[1,-1]
    a,_,_=s.active(p,'no_leadership_gate');assert a.tolist()==[False,False,False,True,False]
    a,_,_=s.active(p,'no_variation_gate');assert a.tolist()==[False,True,False,False,True]
    a,_,_=s.active(p,'binance_leads_upbit');assert a.tolist()==[False,False,False,True,False]
    a,z,_=s.active(p,'one_bar_stale_onset');assert a.tolist()==[False,False,True,False,False] and z[a].tolist()==[1]
    a,z,_=s.active(p,'direction_flip');assert z[a].tolist()==[-1,1]
    a,z,_=s.active(p,'forced_long');assert z[a].eq(1).all()

def test_source_blind_and_hash_bound():
    q=s.UPBIT_QUERY+s.PERP_QUERY;assert 'bars_upbit' in s.UPBIT_QUERY and "KRW-BTC" in s.UPBIT_QUERY and 'bars_binance ' in s.PERP_QUERY and 'funding' not in q.lower() and 'gross9' not in q.lower()
    assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest();v={'한글':'lag'};expected=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert s.canonical_hash(v)==expected
