from __future__ import annotations
import numpy as np
import pandas as pd
from training import build_high_volatility_causal_vixfix_reentry_reversal_support as s

def test_causal_midrank_excludes_current_and_requires_history(monkeypatch) -> None:
    monkeypatch.setitem(s.POLICY,"history_decisions",3);monkeypatch.setitem(s.POLICY,"minimum_history_decisions",2)
    got=s.causal_midrank(pd.Series([1.,2.,3.,2.]))
    assert np.isnan(got.iloc[0]) and np.isnan(got.iloc[1])
    assert got.iloc[2]==1.0
    assert got.iloc[3]==0.5

def test_clock_side_fades_dominant_extreme_displacement(monkeypatch) -> None:
    monkeypatch.setattr(s,"stage_for",lambda *_:"train")
    decisions=pd.to_datetime(["2023-07-01T00:00:00Z","2023-07-02T00:00:00Z"])
    panel=pd.DataFrame({"decision_time":decisions,"feature_available_time":decisions,"eligible":[True,True],"dominant_displacement":[.1,.2],"reversal_side":[1,-1],"displacement_rank":[.8,.9],"realized_variation":[.1,.1],"variation_rank":[.7,.8]})
    clock=s.build_clock(panel)
    assert clock["side"].tolist()==[1,-1]
    assert clock["entry_time"].tolist()==list(panel["decision_time"]+pd.Timedelta("5m"))
    assert clock["exit_time"].tolist()==list(panel["decision_time"]+pd.Timedelta("24h5m"))

def test_panel_requires_contiguous_auctions_and_causal_ranks(monkeypatch) -> None:
    monkeypatch.setitem(s.POLICY,"minimum_history_decisions",1)
    monkeypatch.setitem(s.POLICY,"prior_close_periods",2)
    times=pd.date_range("2023-07-01",periods=5,freq="1d",tz="UTC")
    bars=pd.DataFrame({
        "decision_time":times,"last_close":[100.,101.,102.,100.,101.],
        "auction_high":[101.,102.,103.,122.,103.5],
        "auction_low":[99.,100.,90.,99.,99.],
        "minute_squared_return":[.0004,.0009,.0025,.01,.04],"source_rows":[1440]*5,
        "distinct_rows":[1440]*5,"first_ts":times-pd.Timedelta("24h"),
        "last_ts":times-pd.Timedelta("1m"),"coherent":[True]*5,
    })
    panel=s.build_panel(bars)
    assert panel["source_valid"].tolist()==[False,False,True,True,True]
    assert panel["reversal_side"].tolist()==[0,0,1,-1,-1]
    assert panel["eligible"].tolist()==[False,False,False,False,True]

def test_support_stats_are_exact() -> None:
    clock=pd.DataFrame({"split":["train"]*4,"side":[1,1,-1,-1],"entry_time":pd.to_datetime(["2023-07-01","2023-07-02","2023-08-01","2023-08-02"],utc=True)})
    assert s.support_stats(clock,"train")=={"events":4,"longs":2,"shorts":2,"minority_side_share":.5,"max_month_share":.5}

def test_preregistration_is_hash_bound() -> None:
    assert s.sha256_file(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA
    s.prereg.validate(s.REGISTRATION)
