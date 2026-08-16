import pandas as pd
from training import build_high_volatility_caspc_ehlers_cross_venue_resolution_router_support as s
def test_active_cvdr_overrides_base_side():
 t=pd.Timestamp('2024-01-01T03:05Z');base=pd.DataFrame({'candidate':['b'],'control':['primary'],'split':['test'],'selected_action':['x'],'decision_time':[t-pd.Timedelta(minutes=5)],'feature_available_time':[t-pd.Timedelta(minutes=5)],'entry_time':[t],'exit_time':[t+pd.Timedelta(hours=8)],'side':[1],'active_action_count':[1]});cv=pd.DataFrame({'entry_time':[t-pd.Timedelta(hours=1)],'exit_time':[t+pd.Timedelta(hours=5)],'side':[-1]});out,a=s.route(base,cv);assert out.iloc[0].side==-1 and a['overrides_total']==1
def test_contract():assert s.PREREG_SHA=='a6b24af32d36aad76ebc0ae0b453790234e70a7862a257f9fb2c10886f58f3e1'
