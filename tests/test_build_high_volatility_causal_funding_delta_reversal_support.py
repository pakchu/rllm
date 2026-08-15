from __future__ import annotations
import numpy as np
import pandas as pd
from training import build_high_volatility_causal_funding_delta_reversal_support as s

def test_causal_midrank_excludes_current_and_requires_history(monkeypatch) -> None:
    monkeypatch.setitem(s.POLICY,"history_cycles",3);monkeypatch.setitem(s.POLICY,"minimum_history_cycles",2)
    got=s.causal_midrank(pd.Series([1.,2.,3.,2.]))
    assert np.isnan(got.iloc[0]) and np.isnan(got.iloc[1])
    assert got.iloc[2]==1.0
    assert got.iloc[3]==0.5

def test_clock_side_is_negative_funding_change(monkeypatch) -> None:
    monkeypatch.setattr(s,"stage_for",lambda *_:"train")
    panel=pd.DataFrame({"decision_time":pd.to_datetime(["2023-07-01T00:00:00Z","2023-07-01T08:00:00Z"]),"feature_available_time":pd.to_datetime(["2023-07-01T00:00:00Z","2023-07-01T08:00:00Z"]),"eligible":[True,True],"funding_change":[.001,-.002],"absolute_change_rank":[.8,.9],"realized_variation":[.1,.1],"variation_rank":[.7,.8]})
    clock=s.build_clock(panel)
    assert clock["side"].tolist()==[-1,1]
    assert clock["entry_time"].tolist()==list(panel["decision_time"]+pd.Timedelta("5m"))
    assert clock["exit_time"].tolist()==list(panel["decision_time"]+pd.Timedelta("8h5m"))

def test_support_stats_are_exact() -> None:
    clock=pd.DataFrame({"split":["train"]*4,"side":[1,1,-1,-1],"entry_time":pd.to_datetime(["2023-07-01","2023-07-02","2023-08-01","2023-08-02"],utc=True)})
    assert s.support_stats(clock,"train")=={"events":4,"longs":2,"shorts":2,"minority_side_share":.5,"max_month_share":.5}

def test_preregistration_is_hash_bound() -> None:
    assert s.sha256_file(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA
    s.prereg.validate(s.REGISTRATION)
