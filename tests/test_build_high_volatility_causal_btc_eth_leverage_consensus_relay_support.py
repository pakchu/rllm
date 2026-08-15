from __future__ import annotations
import numpy as np
import pandas as pd
from training import build_high_volatility_causal_btc_eth_leverage_consensus_relay_support as s

def test_causal_midrank_excludes_current_and_requires_history(monkeypatch) -> None:
    monkeypatch.setitem(s.POLICY,"history_cycles",3);monkeypatch.setitem(s.POLICY,"minimum_history_cycles",2)
    got=s.causal_midrank(pd.Series([1.,2.,3.,2.]));assert np.isnan(got.iloc[0]) and np.isnan(got.iloc[1]);assert got.iloc[2]==1. and got.iloc[3]==.5

def test_clock_side_follows_common_btc_eth_direction(monkeypatch) -> None:
    monkeypatch.setattr(s,"stage_for",lambda *_:"train")
    decisions=pd.to_datetime(["2023-07-01T00:00:00Z","2023-07-01T16:00:00Z"])
    panel=pd.DataFrame({"decision_time":decisions,"feature_available_time":decisions,"eligible":[True,True],"oi_change":[.01,.02],"btc_return":[.04,-.05],"eth_return":[.03,-.06],"consensus_displacement":[.03,.05],"consensus_rank":[.8,.9],"realized_variation":[.1,.1],"variation_rank":[.7,.8]})
    clock=s.build_clock(panel);assert clock.side.tolist()==[1,-1]
    assert clock.entry_time.tolist()==list(decisions+pd.Timedelta("5m"));assert clock.exit_time.tolist()==list(decisions+pd.Timedelta("8h5m"))

def test_panel_requires_exact_joint_source_and_causal_ranks(monkeypatch) -> None:
    monkeypatch.setitem(s.POLICY,"minimum_history_cycles",1)
    times=pd.date_range("2023-07-01",periods=5,freq="8h",tz="UTC")
    oi=pd.DataFrame({"decision_time":times,"sum_open_interest":[100.,110.,120.,130.,140.]})
    rows=[]
    btc=[.01,.02,.04,-.06,.08];eth=[.015,.025,.05,.03,.09]
    for symbol,returns in (("BTCUSDT",btc),("ETHUSDT",eth)):
        for t,ret in zip(times,returns):rows.append({"decision_time":t,"symbol":symbol,"completed_return":ret,"minute_squared_return":ret*ret+.001,"source_rows":480,"distinct_rows":480,"first_ts":t-pd.Timedelta("8h"),"last_ts":t-pd.Timedelta("1m"),"coherent":True})
    panel=s.build_panel((oi,pd.DataFrame(rows).sort_values(["decision_time","symbol"]).reset_index(drop=True)))
    assert panel.source_valid.tolist()==[False,True,True,True,True]
    assert panel.eligible.tolist()==[False,False,True,False,True]

def test_support_stats_are_exact() -> None:
    clock=pd.DataFrame({"split":["train"]*4,"side":[1,1,-1,-1],"entry_time":pd.to_datetime(["2023-07-01","2023-07-02","2023-08-01","2023-08-02"],utc=True)})
    assert s.support_stats(clock,"train")=={"events":4,"longs":2,"shorts":2,"minority_side_share":.5,"max_month_share":.5}

def test_preregistration_is_hash_bound() -> None:
    assert s.sha256_file(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA;s.prereg.validate(s.REGISTRATION)
