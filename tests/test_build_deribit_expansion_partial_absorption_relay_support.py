import pandas as pd
from training import build_deribit_expansion_partial_absorption_relay_support as s
def rows():
 t=pd.date_range('2023-08-01T00:00:00Z',periods=9,freq='1h');d=pd.DataFrame({'decision_time':t,'base_valid':True,'bvol_body':-.01,'dvol_body':.02,'first_half_return':.001,'first_q60':.01,'second_half_return':.001});d.loc[1,['first_half_return','second_half_return']]=[-.02,.015];d.loc[8,['first_half_return','second_half_return']]=[.02,-.015];return d
def test_clock_follows_two_sided_partial_absorption():
 c=s.clock(rows());assert list(c.side)==[1,-1];assert list(c.entry_time)==[pd.Timestamp('2023-08-01T01:05:00Z'),pd.Timestamp('2023-08-01T08:05:00Z')]
def test_wrong_vol_polarity_or_same_direction_blocks():
 d=rows();d[['bvol_body','dvol_body']]=d[['bvol_body','dvol_body']]*-1;assert s.clock(d).empty
 d=rows();d.loc[[1,8],'second_half_return']*=-1;assert s.clock(d).empty
def test_schema_and_economics_closed():
 d=rows();d['base_valid']=False;assert list(s.clock(d).columns)==list(s.COLUMNS);assert s.ECONOMIC_OUTCOMES_AUTHORIZED is False
