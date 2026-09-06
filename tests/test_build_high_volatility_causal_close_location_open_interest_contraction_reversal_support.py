from __future__ import annotations
import numpy as np
import pandas as pd
from training import build_high_volatility_causal_close_location_open_interest_contraction_reversal_support as s


def test_causal_midrank_excludes_current(monkeypatch) -> None:
    monkeypatch.setitem(s.POLICY, "history_cycles", 3); monkeypatch.setitem(s.POLICY, "minimum_history_cycles", 2)
    got = s.causal_midrank(pd.Series([1., 2., 3., 2.]))
    assert np.isnan(got.iloc[0]) and np.isnan(got.iloc[1]) and got.iloc[2] == 1. and got.iloc[3] == .5


def test_panel_requires_oi_expansion_outer_close_and_causal_variation(monkeypatch) -> None:
    monkeypatch.setitem(s.POLICY, "minimum_history_cycles", 1)
    times = pd.date_range("2023-07-01", periods=5, freq="8h", tz="UTC")
    oi = pd.DataFrame({"decision_time": times, "sum_open_interest": [130., 120., 125., 110., 100.]})
    bars = pd.DataFrame({"decision_time": times, "final_close": [109., 109., 91., 100., 91.], "cycle_high": [110.]*5, "cycle_low": [90.]*5, "completed_return": [.01,.02,-.03,.04,-.05], "minute_squared_return": [.001,.002,.003,.004,.005], "source_rows": [480]*5, "distinct_rows": [480]*5, "first_ts": times-pd.Timedelta("8h"), "last_ts": times-pd.Timedelta("1m"), "coherent": [True]*5})
    panel=s.build_panel((oi,bars))
    assert panel.source_valid.tolist() == [False,True,True,True,True]
    assert panel.eligible.tolist() == [False,False,False,False,True]


def test_clock_follows_completed_direction(monkeypatch) -> None:
    monkeypatch.setattr(s,"stage_for",lambda *_:"train")
    d=pd.to_datetime(["2023-07-01T00:00:00Z","2023-07-01T16:00:00Z"])
    p=pd.DataFrame({"decision_time":d,"feature_available_time":d,"eligible":[True,True],"oi_change":[.1,.2],"close_location":[.9,.1],"completed_return":[.02,-.03],"realized_variation":[.1,.2],"variation_rank":[.8,.9]})
    c=s.build_clock(p); assert c.side.tolist()==[-1,1]


def test_preregistration_is_hash_bound() -> None:
    assert s.sha256_file(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA; s.prereg.validate(s.REGISTRATION)
