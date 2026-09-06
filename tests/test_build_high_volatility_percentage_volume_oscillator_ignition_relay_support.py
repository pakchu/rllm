import hashlib

import numpy as np
import pandas as pd

from training import build_high_volatility_percentage_volume_oscillator_ignition_relay_support as s


def test_cross_up_and_pvo_formula(monkeypatch):
    assert s.cross_up(pd.Series([-1.0, 1.0, 2.0, -1.0, 1.0])).tolist() == [False, True, False, False, True]
    monkeypatch.setitem(s.P, "fast_periods", 2)
    monkeypatch.setitem(s.P, "slow_periods", 3)
    monkeypatch.setitem(s.P, "signal_periods", 2)
    values=s.pvo_values(pd.Series([1.,2.,3.,4.,5.,6.,7.]),pd.Series([True]*7))
    assert np.isnan(values.pvo.iloc[1]) and np.isnan(values.difference.iloc[2])
    assert values.pvo.iloc[-1]>0 and np.isfinite(values.difference.iloc[-1])


def test_prior_rank_excludes_current(monkeypatch):
    monkeypatch.setitem(s.P,"minimum_variation_history_decisions",2)
    monkeypatch.setitem(s.P,"variation_history_decisions",3)
    rank=s.prior_rank(pd.Series([1.,2.,np.nan,3.]));assert np.isnan(rank.iloc[1]) and rank.iloc[3]==1


def panel():
    return pd.DataFrame({"source_valid":[True]*6,"entry_side":[-1,1,1,-1,-1,1],"signal_ignition":[False,False,True,False,True,False],"zero_line_ignition":[False,False,True,False,False,True],"variation_rank":[.8,.8,.8,.8,.4,.8]})


def test_controls():
    active,side,_=s.active(panel());assert active.tolist()==[False,False,True,False,False,False] and side[active].tolist()==[1]
    assert s.active(panel(),"no_variation_gate")[0].iloc[4]
    zero,zero_side,_=s.active(panel(),"zero_line_ignition");assert zero.iloc[2] and zero.iloc[5] and zero_side[zero].tolist()==[1,1]
    stale,stale_side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and stale_side.iloc[3]==1
    forced,forced_side,_=s.active(panel(),"forced_long");assert forced_side[forced].eq(1).all()


def test_source_is_blind_and_hash_bound():
    assert "funding" not in s.QUERY.lower() and "gross9" not in s.QUERY.lower()
    assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
