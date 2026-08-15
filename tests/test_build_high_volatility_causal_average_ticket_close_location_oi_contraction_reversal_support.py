from __future__ import annotations
import numpy as np
import pandas as pd
from training import build_high_volatility_causal_average_ticket_close_location_oi_contraction_reversal_support as s

def test_causal_midrank_excludes_current_and_requires_history(monkeypatch) -> None:
    monkeypatch.setitem(s.POLICY,"history_cycles",3);monkeypatch.setitem(s.POLICY,"minimum_history_cycles",2)
    got=s.causal_midrank(pd.Series([1.,2.,3.,2.]));assert np.isnan(got.iloc[0]) and np.isnan(got.iloc[1]);assert got.iloc[2]==1. and got.iloc[3]==.5

def test_clock_side_follows_completed_return(monkeypatch) -> None:
    monkeypatch.setattr(s,"stage_for",lambda *_:"train")
    decisions=pd.to_datetime(["2023-07-01T00:00:00Z","2023-07-01T16:00:00Z"])
    panel=pd.DataFrame({"decision_time":decisions,"feature_available_time":decisions,"eligible":[True,True],"oi_change":[.01,.02],"first_half_quote_turnover":[1000.,1200.],"second_half_quote_turnover":[1800.,2200.],"first_half_trade_count":[100.,100.],"second_half_trade_count":[150.,180.],"first_half_average_ticket":[10.,12.],"second_half_average_ticket":[12.,12.222222],"close_location":[.9,.1],"completed_return":[.04,-.05],"average_ticket_acceleration":[.1823,.01835],"average_ticket_acceleration_rank":[.8,.9],"realized_variation":[.1,.1],"variation_rank":[.7,.8]})
    clock=s.build_clock(panel);assert clock.side.tolist()==[-1,1]
    assert clock.entry_time.tolist()==list(decisions+pd.Timedelta("5m"));assert clock.exit_time.tolist()==list(decisions+pd.Timedelta("8h5m"))

def test_panel_requires_ticket_acceleration_close_location_and_causal_ranks(monkeypatch) -> None:
    monkeypatch.setitem(s.POLICY,"minimum_history_cycles",1)
    times=pd.date_range("2023-07-01",periods=5,freq="8h",tz="UTC")
    oi=pd.DataFrame({"decision_time":times,"sum_open_interest":[140.,130.,120.,110.,100.]})
    bars=pd.DataFrame({"decision_time":times,"first_quote_turnover":[1000.]*5,"first_trade_count":[100.]*5,"second_quote_turnover":[1100.,1320.,1680.,1600.,3000.],"second_trade_count":[110.,120.,140.,160.,200.],"final_close":[109.,101.,91.,109.,109.],"cycle_high":[110.]*5,"cycle_low":[90.]*5,"completed_return":[.01,.02,-.04,0.,.08],"minute_squared_return":[.001,.002,.003,.004,.005],"source_rows":[480]*5,"distinct_rows":[480]*5,"first_ts":times-pd.Timedelta("8h"),"last_ts":times-pd.Timedelta("1m"),"coherent":[True]*5})
    panel=s.build_panel((oi,bars));assert panel.source_valid.tolist()==[False,True,True,False,True]
    assert panel.eligible.tolist()==[False,False,True,False,True]

def test_support_stats_are_exact() -> None:
    clock=pd.DataFrame({"split":["train"]*4,"side":[1,1,-1,-1],"entry_time":pd.to_datetime(["2023-07-01","2023-07-02","2023-08-01","2023-08-02"],utc=True)})
    assert s.support_stats(clock,"train")=={"events":4,"longs":2,"shorts":2,"minority_side_share":.5,"max_month_share":.5}

def test_preregistration_is_hash_bound() -> None:
    assert s.sha256_file(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA;s.prereg.validate(s.REGISTRATION)
