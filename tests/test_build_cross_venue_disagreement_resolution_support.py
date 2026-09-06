import pandas as pd
from training import build_cross_venue_disagreement_resolution_support as s
def rows():
 t=pd.date_range('2023-08-01T00:00:00Z',periods=9,freq='1h');d=pd.DataFrame({'decision_time':t,'base_valid':True,'bvol_body':.01,'dvol_body':-.02,'hour_return':.001,'q40':.004,'q75':.02});d.loc[1,'hour_return']=.01;d.loc[8,'hour_return']=-.01;return d
def test_clock_follows_two_sided_moderate_resolution():
 c=s.clock(rows());assert list(c.side)==[1,-1];assert list(c.entry_time)==[pd.Timestamp('2023-08-01T01:05:00Z'),pd.Timestamp('2023-08-01T08:05:00Z')]
def test_same_sign_vol_or_tail_return_blocks():
 d=rows();d['dvol_body']=.02;assert s.clock(d).empty
 d=rows();d.loc[[1,8],'hour_return']=.03;assert s.clock(d).empty
def test_schema_and_economics_closed():
 d=rows();d['base_valid']=False;assert list(s.clock(d).columns)==list(s.COLUMNS);assert s.ECONOMIC_OUTCOMES_AUTHORIZED is False
