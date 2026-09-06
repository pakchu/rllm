import hashlib,json
import numpy as np
import pandas as pd
from training import build_high_volatility_variance_signature_reversal_relay_support as s


def test_exact_multiscale_close_to_close_signature(monkeypatch):
    start=pd.Timestamp('2023-04-01T00:00:00Z');end=start+pd.Timedelta('24h')
    monkeypatch.setattr(s,'START',start);monkeypatch.setattr(s,'END',end)
    index=pd.date_range(start,end,freq='1min',inclusive='left');r=.0001;close=np.exp(np.arange(len(index))*r)
    source=pd.DataFrame({'close':close,'row_valid':True},index=index)
    states=s.multiscale_states(source);last=states.iloc[-1]
    assert np.isclose(last.one_minute_rv,1439*r*r)
    assert np.isclose(last.five_minute_rv,287*(5*r)**2)
    assert np.isclose(last.one_minute_rv/last.five_minute_rv,1439/(287*25))
    assert np.isclose(last.last_five_minute_return,5*r)


def test_multiscale_gap_requires_full_rewarm(monkeypatch):
    start=pd.Timestamp('2023-04-01T00:00:00Z');end=start+pd.Timedelta('28h')
    monkeypatch.setattr(s,'START',start);monkeypatch.setattr(s,'END',end)
    index=pd.date_range(start,end,freq='1min',inclusive='left');close=np.exp(np.arange(len(index))*.0001);valid=np.ones(len(index),dtype=bool);valid[100]=False
    states=s.multiscale_states(pd.DataFrame({'close':close,'row_valid':valid},index=index))
    assert states.one_minute_rv.iloc[5] != states.one_minute_rv.iloc[5]
    assert np.isfinite(states.one_minute_rv.iloc[-1])


def test_prior_rank_excludes_current_and_resets(monkeypatch):
    monkeypatch.setitem(s.P,'minimum_rank_history_decisions',2);monkeypatch.setitem(s.P,'rank_history_decisions',3)
    ranks=s.prior_rank(pd.Series([1.,2.,3.,np.nan,4.,5.,6.]),pd.Series([True,True,True,False,True,True,True]));assert np.isnan(ranks.iloc[1]) and ranks.iloc[2]==1 and np.isnan(ranks.iloc[5]) and ranks.iloc[6]==1


def test_onset_requires_prior_valid_and_frozen_controls():
    assert s.fresh_onset(pd.Series([True,False,True,True,False,True]),pd.Series([True,True,True,False,True,True])).tolist()==[False,False,True,False,False,True]
    p=pd.DataFrame({'source_valid':[True]*5,'signature_tail':[False,True,True,False,True],'variation_tail':[True,True,False,True,True],'eligible':[False,True,False,False,True],'entry_side':[-1,1,-1,1,-1],'last_five_minute_return':[1.,-1.,1.,-1.,1.]})
    a,z,_=s.active(p);assert a.tolist()==[False,True,False,False,True] and z[a].tolist()==[1,-1]
    a,z,_=s.active(p,'no_signature_gate');assert a.tolist()==[False,False,False,True,False]
    a,z,_=s.active(p,'no_variation_gate');assert a.tolist()==[False,True,False,False,True]
    a,z,_=s.active(p,'five_minute_return_continuation');assert z[a].tolist()==[-1,1]
    a,z,_=s.active(p,'one_bar_stale_onset');assert a.tolist()==[False,False,True,False,False] and z[a].tolist()==[1]
    a,z,_=s.active(p,'direction_flip');assert z[a].tolist()==[-1,1]
    a,z,_=s.active(p,'forced_long');assert z[a].eq(1).all()


def test_source_blind_and_hash_bound():
    q=s.QUERY.lower();assert 'open,high,low,close' in q and 'funding' not in q and 'gross9' not in q
    assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
    v={'한글':'signature'};expected=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert s.canonical_hash(v)==expected
