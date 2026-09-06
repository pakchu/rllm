import pandas as pd
from training import build_high_volatility_cross_structure_action_vote_oi_sponsorship_support as s
def test_oi_gate_uses_exact_completed_8h_growth():
 t=pd.Timestamp('2024-01-01T00:00Z');base=pd.DataFrame({'candidate':['x'],'control':['primary'],'split':['test'],'eligibility_id':['e'],'decision_time':[t],'feature_available_time':[t],'entry_time':[t+pd.Timedelta(minutes=5)],'exit_time':[t+pd.Timedelta(hours=8,minutes=5)],'side':[1],'active_action_count':[2],'long_vote_count':[2],'short_vote_count':[0]});oi=pd.DataFrame({'ts':[t,t+pd.Timedelta(hours=8)],'sum_open_interest':[100.,110.]});out=s.apply_gate(base,oi);assert len(out)==1 and out.iloc[0].oi_change>0
def test_contract():assert s.PREREG_SHA=='15699ae93b1d2cd8e44435f3376f977302c0ef63d44e8205feb20155abe38530'
