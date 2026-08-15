from __future__ import annotations
import numpy as np
import pandas as pd
from training import build_high_volatility_spot_lead_open_interest_acceleration_continuation_support as s

def test_causal_midrank_excludes_current(monkeypatch):
    monkeypatch.setitem(s.POLICY,'history_cycles',3);monkeypatch.setitem(s.POLICY,'minimum_history_cycles',2)
    got=s.causal_midrank(pd.Series([1.,2.,3.,2.]));assert np.isnan(got.iloc[0]) and np.isnan(got.iloc[1]) and got.iloc[2]==1. and got.iloc[3]==.5

def test_panel_requires_oi_expansion_agreement_spot_lead_and_variation(monkeypatch):
    monkeypatch.setitem(s.POLICY,'minimum_history_cycles',1)
    t=pd.date_range('2023-07-01',periods=5,freq='8h',tz='UTC')
    oi=pd.DataFrame({'decision_time':t,'sum_open_interest':[100.,120.,115.,130.,140.]})
    bars=pd.DataFrame({'decision_time':t,'perpetual_return':[.005,.01,-.02,.04,-.03],'spot_return':[.01,.02,-.03,-.02,-.05],'minute_squared_return':[.001,.002,.003,.004,.005],'source_rows':[480]*5,'distinct_rows':[480]*5,'first_ts':t-pd.Timedelta('8h'),'last_ts':t-pd.Timedelta('1m'),'coherent':[True]*5})
    panel=s.build_panel((oi,bars));assert panel.source_valid.tolist()==[False,True,True,True,True];assert panel.eligible.tolist()==[False,False,False,False,True]

def test_clock_follows_spot_direction(monkeypatch):
    monkeypatch.setattr(s,'stage_for',lambda *_:'train');d=pd.to_datetime(['2023-07-01T00:00:00Z','2023-07-01T16:00:00Z'])
    p=pd.DataFrame({'decision_time':d,'feature_available_time':d,'eligible':[True,True],'oi_change':[.1,.2],'perpetual_return':[.01,-.02],'spot_return':[.02,-.03],'directional_overshoot':[.01,.01],'realized_variation':[.1,.2],'variation_rank':[.8,.9]})
    assert s.build_clock(p).side.tolist()==[1,-1]

def test_preregistration_is_hash_bound():
    assert s.sha256_file(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA;s.prereg.validate(s.REGISTRATION)
