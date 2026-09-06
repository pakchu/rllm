import pandas as pd
from training import build_realized_over_implied_volatility_shock_continuation_relay_support as s
def test_clock_requires_ratio_tail_and_direction_confirmation():
 t=pd.date_range('2023-08-01',periods=3,freq='8h',tz='UTC');x=pd.DataFrame({'decision_time':t,'realized_variation':.1,'implied_level':.5,'ratio_log':-1.,'ratio_rank':[.8,.8,.8],'realized_rank':.8,'implied_rank':.2,'block_return':[.01,.01,-.01],'final_hour_return':[.001,-.001,-.001]});c=s.clock(x);assert list(c.side)==[1,-1]
def test_empty_schema_and_economics_closed():
 x=pd.DataFrame({k:[] for k in ['decision_time','realized_variation','implied_level','ratio_log','ratio_rank','realized_rank','implied_rank','block_return','final_hour_return']});x['decision_time']=pd.to_datetime(x.decision_time,utc=True);assert list(s.clock(x).columns)==list(s.COLUMNS) and s.ECONOMIC_OUTCOMES_AUTHORIZED is False
