from __future__ import annotations
import numpy as np
import pandas as pd
from training import build_high_volatility_causal_variance_open_interest_acceleration_relay_support as s

def test_causal_midrank_excludes_current(monkeypatch):
    monkeypatch.setitem(s.POLICY,"history_cycles",3);monkeypatch.setitem(s.POLICY,"minimum_history_cycles",2)
    x=s.causal_midrank(pd.Series([1.,2.,3.,2.]));assert np.isnan(x.iloc[0]) and np.isnan(x.iloc[1]) and x.iloc[2]==1. and x.iloc[3]==.5

def test_panel_requires_variance_acceleration_oi_and_ranks(monkeypatch):
    monkeypatch.setitem(s.POLICY,"minimum_history_cycles",1)
    t=pd.date_range("2023-07-01",periods=5,freq="8h",tz="UTC")
    oi=pd.DataFrame({"decision_time":t,"sum_open_interest":[100.,105.,120.,115.,140.]})
    bars=pd.DataFrame({"decision_time":t,"first_variance":[1.,1.,2.,1.,1.],"second_variance":[2.,3.,4.,3.,5.],"completed_return":[.01,.02,-.03,.04,-.05],"minute_squared_return":[3.,4.,6.,4.,6.],"source_rows":[480]*5,"distinct_rows":[480]*5,"first_ts":t-pd.Timedelta("8h"),"last_ts":t-pd.Timedelta("1m"),"coherent":[True]*5})
    p=s.build_panel((oi,bars));assert p.source_valid.tolist()==[False,False,True,True,True];assert p.eligible.iloc[4]

def test_clock_follows_completed_return(monkeypatch):
    monkeypatch.setattr(s,"stage_for",lambda *_:"train");d=pd.to_datetime(["2023-07-01T00:00:00Z","2023-07-01T16:00:00Z"])
    p=pd.DataFrame({"decision_time":d,"feature_available_time":d,"eligible":[True,True],"prior_oi_growth":[.01,.02],"current_oi_growth":[.1,.2],"oi_growth_acceleration":[.09,.18],"first_half_variance":[1.,1.],"second_half_variance":[2.,3.],"completed_return":[.02,-.03],"variance_acceleration":[.69,1.1],"variance_acceleration_rank":[.8,.9],"realized_variation":[.1,.2],"variation_rank":[.8,.9]})
    assert s.build_clock(p).side.tolist()==[1,-1]

def test_preregistration_hash_bound():
    assert s.sha256_file(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA;s.prereg.validate(s.REGISTRATION)
