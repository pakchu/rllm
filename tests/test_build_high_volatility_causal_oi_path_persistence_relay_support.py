from __future__ import annotations
import numpy as np
import pandas as pd
from training import build_high_volatility_causal_oi_path_persistence_relay_support as s

def test_causal_midrank_excludes_current_and_requires_history(monkeypatch) -> None:
    monkeypatch.setitem(s.POLICY,"history_cycles",3);monkeypatch.setitem(s.POLICY,"minimum_history_cycles",2)
    got=s.causal_midrank(pd.Series([1.,2.,3.,2.]))
    assert np.isnan(got.iloc[0]) and np.isnan(got.iloc[1])
    assert got.iloc[2]==1.0
    assert got.iloc[3]==0.5

def test_clock_side_follows_completed_efficient_path(monkeypatch) -> None:
    monkeypatch.setattr(s,"stage_for",lambda *_:"train")
    decisions=pd.to_datetime(["2023-07-01T00:00:00Z","2023-07-02T00:00:00Z"])
    panel=pd.DataFrame({"decision_time":decisions,"feature_available_time":decisions,"eligible":[True,True],"oi_change":[.01,.02],"completed_return":[.04,-.05],"path_efficiency":[.7,.8],"efficiency_rank":[.8,.9],"realized_variation":[.1,.1],"variation_rank":[.7,.8]})
    clock=s.build_clock(panel)
    assert clock["side"].tolist()==[1,-1]
    assert clock["entry_time"].tolist()==list(panel["decision_time"]+pd.Timedelta("5m"))
    assert clock["exit_time"].tolist()==list(panel["decision_time"]+pd.Timedelta("24h5m"))

def test_panel_requires_exact_gap_and_causal_ranks(monkeypatch) -> None:
    monkeypatch.setitem(s.POLICY,"minimum_history_cycles",1)
    times=pd.date_range("2023-07-01",periods=5,freq="8h",tz="UTC")
    oi=pd.DataFrame({"decision_time":times,"sum_open_interest":[100.,110.,120.,130.,140.]})
    bars=pd.DataFrame({
        "decision_time":times,
        "first_half_return":[.005,.01,.02,-.03,.04],
        "second_half_return":[.005,.01,.02,-.03,-.04],
        "completed_return":[.01,.02,.04,-.06,0.],
        "absolute_path_return":[.1]*5,
        "minute_squared_return":[.0004,.0009,.0025,.01,.04],"source_rows":[480]*5,
        "distinct_rows":[480]*5,"first_ts":times-pd.Timedelta("8h"),
        "last_ts":times-pd.Timedelta("1m"),"coherent":[True]*5,
    })
    panel=s.build_panel((oi,bars))
    assert panel["source_valid"].tolist()==[False,True,True,True,False]
    assert panel["eligible"].tolist()==[False,False,True,True,False]

def test_support_stats_are_exact() -> None:
    clock=pd.DataFrame({"split":["train"]*4,"side":[1,1,-1,-1],"entry_time":pd.to_datetime(["2023-07-01","2023-07-02","2023-08-01","2023-08-02"],utc=True)})
    assert s.support_stats(clock,"train")=={"events":4,"longs":2,"shorts":2,"minority_side_share":.5,"max_month_share":.5}

def test_preregistration_is_hash_bound() -> None:
    assert s.sha256_file(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA
    s.prereg.validate(s.REGISTRATION)
