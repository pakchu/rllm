import pandas as pd
from training import build_deribit_led_shock_deceleration_reversal_support as s
def rows():
 t=pd.date_range('2023-08-01T00:00:00Z',periods=9,freq='1h');d=pd.DataFrame({'decision_time':t,'base_valid':True,'bvol_body':.01,'dvol_body':-.02,'first_half_return':.001,'first_q75':.02,'second_half_return':-.001});d.loc[1,['first_half_return','second_half_return']]=[-.04,-.01];d.loc[8,['first_half_return','second_half_return']]=[.04,.01];return d
def test_clock_fades_two_sided_deribit_led_deceleration():
 c=s.clock(rows());assert list(c.side)==[1,-1];assert list(c.entry_time)==[pd.Timestamp('2023-08-01T01:05:00Z'),pd.Timestamp('2023-08-01T08:05:00Z')]
def test_binance_led_or_opposite_second_half_blocks():
 d=rows();d['dvol_body']=-.005;assert s.clock(d).empty
 d=rows();d.loc[[1,8],'second_half_return']*=-1;assert s.clock(d).empty
def test_schema_and_economics_closed():
 d=rows();d['base_valid']=False;assert list(s.clock(d).columns)==list(s.COLUMNS);assert s.ECONOMIC_OUTCOMES_AUTHORIZED is False
