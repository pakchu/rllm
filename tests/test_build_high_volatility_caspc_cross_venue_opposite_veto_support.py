import pandas as pd
from training import build_high_volatility_caspc_cross_venue_opposite_veto_support as s
def base_row(t):
 return pd.DataFrame({'candidate':['b'],'control':['primary'],'split':['test'],'selected_action':['x'],'decision_time':[t-pd.Timedelta(minutes=5)],'feature_available_time':[t-pd.Timedelta(minutes=5)],'entry_time':[t],'exit_time':[t+pd.Timedelta(hours=8)],'side':[1],'active_action_count':[1]})
def test_active_opposite_cvdr_vetoes_event():
 t=pd.Timestamp('2024-01-01T03:05Z');cv=pd.DataFrame({'entry_time':[t-pd.Timedelta(hours=1)],'exit_time':[t+pd.Timedelta(hours=5)],'side':[-1]});out,a=s.route(base_row(t),cv);assert out.empty and a['opposite_vetoes_total']==1
def test_active_same_side_cvdr_keeps_base_side():
 t=pd.Timestamp('2024-01-01T03:05Z');cv=pd.DataFrame({'entry_time':[t-pd.Timedelta(hours=1)],'exit_time':[t+pd.Timedelta(hours=5)],'side':[1]});out,a=s.route(base_row(t),cv);assert len(out)==1 and out.iloc[0].side==1 and a['same_side_total']==1
def test_contract():assert s.PREREG_SHA=='c02528f477e2c1264a9a6f926c04d8eac032c9f5d6b0b5948b908755bed3a21b'
