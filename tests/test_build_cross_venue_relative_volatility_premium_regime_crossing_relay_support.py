import numpy as np,pandas as pd
from training import build_cross_venue_relative_volatility_premium_regime_crossing_relay_support as s

def test_causal_midrank_excludes_current():
 x=pd.Series(list(range(672))+[1000.]);r=s.causal_midrank(x);assert np.isnan(r.iloc[671]) and r.iloc[672]==1

def test_crossing_maps_low_long_high_short_and_reserves():
 t=pd.date_range('2023-08-01',periods=16,freq='1h',tz='UTC');r=[.5,.9,.9]+[.5]*11+[.1,.1];d=pd.DataFrame({'decision_time':t,'base_valid':True,'relative_log_level':0.,'relative_rank':r,'bvol_close':40.,'bvol_rank':[.5]*16,'dvol_close':42.,'dvol_rank':[.5]*16})
 c=s.clock(d);assert list(c.side)==[-1,1];assert list(c.entry_time)==[t[1]+pd.Timedelta(minutes=5),t[14]+pd.Timedelta(minutes=5)]

def test_empty_schema_and_economics_closed():
 d=pd.DataFrame({'decision_time':pd.to_datetime([],utc=True),'base_valid':pd.Series([],dtype=bool),'relative_log_level':[],'relative_rank':[],'bvol_close':[],'bvol_rank':[],'dvol_close':[],'dvol_rank':[]});assert list(s.clock(d).columns)==list(s.COLUMNS) and s.ECONOMIC_OUTCOMES_AUTHORIZED is False
