from __future__ import annotations
import numpy as np
import pandas as pd
from training import build_high_volatility_causal_open_interest_acceleration_relay_support as s

def test_causal_midrank_excludes_current_and_requires_history(monkeypatch) -> None:
    monkeypatch.setitem(s.POLICY,"history_cycles",3);monkeypatch.setitem(s.POLICY,"minimum_history_cycles",2)
    got=s.causal_midrank(pd.Series([1.,2.,3.,2.]));assert np.isnan(got.iloc[0]) and np.isnan(got.iloc[1]);assert got.iloc[2]==1. and got.iloc[3]==.5

def test_clock_side_follows_completed_return(monkeypatch) -> None:
    monkeypatch.setattr(s,"stage_for",lambda *_:"train")
    decisions=pd.to_datetime(["2023-07-01T00:00:00Z","2023-07-01T16:00:00Z"])
    panel=pd.DataFrame({"decision_time":decisions,"feature_available_time":decisions,"eligible":[True,True],"start_open_interest":[100.,120.],"midpoint_open_interest":[110.,130.],"sum_open_interest":[120.,150.],"first_half_oi_change":[.095,.08],"second_half_oi_change":[.087,.143],"oi_change":[.182,.223],"oi_acceleration":[.01,.06],"oi_acceleration_rank":[.8,.9],"completed_return":[.04,-.05],"realized_variation":[.1,.1],"variation_rank":[.7,.8]})
    clock=s.build_clock(panel);assert clock.side.tolist()==[1,-1]
    assert clock.entry_time.tolist()==list(decisions+pd.Timedelta("5m"));assert clock.exit_time.tolist()==list(decisions+pd.Timedelta("8h5m"))

def test_panel_requires_positive_oi_acceleration_and_causal_ranks(monkeypatch) -> None:
    monkeypatch.setitem(s.POLICY,"minimum_history_cycles",1)
    times=pd.date_range("2023-07-01",periods=13,freq="4h",tz="UTC")
    log_oi=np.array([0.,.01,.03,.04,.07,.11,.12,.14,.17,.18,.22,.27,.28])
    oi=pd.DataFrame({"decision_time":times,"sum_open_interest":100*np.exp(log_oi)})
    decisions=times[[2,4,6,8,10,12]]
    bars=pd.DataFrame({"decision_time":decisions,"completed_return":[.01,.02,-.04,.03,.08,-.02],"minute_squared_return":[.001,.002,.003,.004,.005,.006],"source_rows":[480]*6,"distinct_rows":[480]*6,"first_ts":decisions-pd.Timedelta("8h"),"last_ts":decisions-pd.Timedelta("1m"),"coherent":[True]*6})
    panel=s.build_panel((oi,bars))
    assert panel.source_valid.tolist()==[False,True,True,False,True,True,False]
    assert panel.eligible.iloc[1] == False
    assert panel.eligible.iloc[2] == True
    assert panel.loc[panel.source_valid,"oi_acceleration"].gt(0).all()

def test_support_stats_are_exact() -> None:
    clock=pd.DataFrame({"split":["train"]*4,"side":[1,1,-1,-1],"entry_time":pd.to_datetime(["2023-07-01","2023-07-02","2023-08-01","2023-08-02"],utc=True)})
    assert s.support_stats(clock,"train")=={"events":4,"longs":2,"shorts":2,"minority_side_share":.5,"max_month_share":.5}

def test_preregistration_is_hash_bound() -> None:
    assert s.sha256_file(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA;s.prereg.validate(s.REGISTRATION)
