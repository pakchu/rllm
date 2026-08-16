import pandas as pd
from training import build_high_volatility_caspc_ehlers_cash_side_router_support as s
def test_cash_return_has_side_authority():
 t=pd.Timestamp('2024-01-01T03:00Z');base=pd.DataFrame({'candidate':['b'],'control':['primary'],'split':['test'],'selected_action':['x'],'decision_time':[t],'feature_available_time':[t],'entry_time':[t+pd.Timedelta(minutes=5)],'exit_time':[t+pd.Timedelta(hours=8,minutes=5)],'side':[1],'active_action_count':[1]});spot=pd.DataFrame({'decision_time':[t],'source_valid':[True],'cash_return':[-.01]});out=s.route(base,spot);assert out.iloc[0].base_side==1 and out.iloc[0].side==-1
def test_contract():assert s.PREREG_SHA=='f5ba5f811001ee26bee38579d8528e142ab1ddfbdf8cc3e0f64efcdee615667f'
