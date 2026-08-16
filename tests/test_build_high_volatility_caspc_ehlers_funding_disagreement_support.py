import pandas as pd
from training import build_high_volatility_caspc_ehlers_funding_disagreement_support as s
def test_gate_uses_latest_causal_opposite_funding():
 t=pd.Timestamp('2024-01-01T03:00Z');base=pd.DataFrame({'candidate':['x','x'],'control':['primary','primary'],'split':['test','test'],'decision_time':[t,t+pd.Timedelta(hours=8)],'feature_available_time':[t,t+pd.Timedelta(hours=8)],'entry_time':[t+pd.Timedelta(minutes=5),t+pd.Timedelta(hours=8,minutes=5)],'exit_time':[t+pd.Timedelta(hours=8,minutes=5),t+pd.Timedelta(hours=16,minutes=5)],'side':[1,-1]});fund=pd.DataFrame({'funding_time':[t-pd.Timedelta(hours=3),t+pd.Timedelta(hours=5)],'funding_rate':[-.001,-.001]});out=s.apply_gate(base,fund);assert len(out)==1 and out.iloc[0].side==1
def test_frozen_contract():assert s.PREREG_SHA=='0145fffe2fb89c32822cec1cf0cea427400eeb43946d4d138aed14168feb37e5' and s.MINIMUM_EVENTS=={'train':8,'test':12,'eval':12,'final':8}
