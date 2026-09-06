import numpy as np,pandas as pd
from training import build_high_volatility_price_flow_open_interest_concordance_router_support as s
def test_causal_midrank_excludes_current(monkeypatch):
 monkeypatch.setitem(s.POLICY,'history_cycles',3);monkeypatch.setitem(s.POLICY,'minimum_history_cycles',2);g=s.causal_midrank(pd.Series([1.,2.,3.,2.]));assert np.isnan(g.iloc[0]) and np.isnan(g.iloc[1]) and g.iloc[2]==1. and g.iloc[3]==.5
def test_clock_uses_flow_direction(monkeypatch):
 monkeypatch.setattr(s,'stage_for',lambda *_:'train');d=pd.to_datetime(['2023-07-01T00:00:00Z','2023-07-01T16:00:00Z']);p=pd.DataFrame({'decision_time':d,'feature_available_time':d,'eligible':[True,True],'oi_change':[.1,-.2],'price_return':[.02,.03],'flow_share':[.1,-.1],'realized_variation':[.1,.2],'variation_rank':[.8,.9]});assert s.build_clock(p).side.tolist()==[1,-1]
def test_prereg_hash_bound():assert s.sha256_file(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA;s.prereg.validate(s.REGISTRATION)
