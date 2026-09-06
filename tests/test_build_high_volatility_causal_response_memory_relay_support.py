import numpy as np,pandas as pd
from training import build_high_volatility_causal_response_memory_relay_support as support
def sample():
 d=pd.date_range('2023-01-01',periods=20,freq='12h',tz='UTC');return pd.DataFrame({'decision_time':d,'entry_time':d+pd.Timedelta(minutes=5),'response_available_time':d+pd.Timedelta(hours=12,minutes=5),'shock_return':[.01]*20,'range_vol':[.02]*20,'signed_response':[.001]*20,'eligible_opportunity':[True]*20})
def test_memory_uses_only_responses_available_by_decision():
 s=support.causal_state(sample());assert not s.active.iloc[:17].any();assert s.active.iloc[17];assert s.memory_count.iloc[17]==16;assert s.side.iloc[17]==1
def test_fixed_controls_and_flip():
 s=sample();m=support.causal_state(s,'fixed_momentum');r=support.causal_state(s,'fixed_reversal');f=support.causal_state(s,'direction_flip');assert m.side.eq(1).all();assert r.side.eq(-1).all();assert f.loc[f.active,'side'].eq(-1).all()
def test_bindings_and_metrics_sealed():
 assert support.sha256(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA;src=support.Path(support.__file__).read_text();assert '"stage_economic_metrics_opened":False' in src;assert '"gross9_rows_opened":False' in src
