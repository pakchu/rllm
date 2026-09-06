import pandas as pd
from training import build_nasdaq_volatility_rotation_btc_confirmation_relay_support as s
def rows():
 return pd.DataFrame({'valid':[True,True,True],'delta_log_relative':[.1,.1,-.1],'absolute_rotation_rank':[.7,.7,.7],'delta_log_vxn':[.1,.1,-.1],'absolute_vxn_rank':[.7]*3,'delta_log_vix':[.05,.05,-.05],'absolute_vix_rank':[.7]*3,'overnight_btc_return':[-.01,.01,.01]})
def test_primary_requires_rotation_and_btc_confirmation():assert list(s.signal(rows(),'primary'))==[-1,0,1]
def test_direction_flip_is_diagnostic_only():assert list(s.signal(rows(),'direction_flip'))==[1,0,-1]
def test_empty_clock_schema_and_economics_closed():
 d=rows().iloc[:0].copy();d['observation_date']=pd.Series(dtype=object);d['next_source_date']=pd.Series(dtype=object);d['overnight_start_time']=pd.to_datetime([],utc=True);d['decision_time']=pd.to_datetime([],utc=True);assert list(s.build_clock(d).columns)==list(s.COLUMNS) and s.ECONOMIC_OUTCOMES_AUTHORIZED is False
