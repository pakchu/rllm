import numpy as np
import pandas as pd
from training import build_confirmation_ladder_reversal_completion_relay_support as support

def _minutes(start,periods):
 t=pd.date_range(start,periods=periods,freq="1min");c=np.linspace(100,110,periods);o=np.r_[100.,c[:-1]];return pd.DataFrame({"ts":t,"open":o,"high":np.maximum(o,c)+.1,"low":np.minimum(o,c)-.1,"close":c,"duplicate_count":1})

def test_ceil():
 assert support.ceil_5m(301)==pd.Timestamp(600,unit="s",tz="UTC")

def test_interval_return():
 start=pd.Timestamp("2024-01-01T00:00:00Z");m=support.prepare_minutes(_minutes(start,10));v=support.interval_return(m,int(start.timestamp()),int((start+pd.Timedelta(minutes=10)).timestamp()));assert v is not None and v[0]>0 and v[1]==10

def test_invalid_duration():
 start=pd.Timestamp("2024-01-01T00:00:00Z");m=support.prepare_minutes(_minutes(start,40));assert support.interval_return(m,int(start.timestamp()),int((start+pd.Timedelta(minutes=31)).timestamp())) is None

def test_primary_and_flip():
 f=pd.DataFrame({"source_valid":[True,True],"eligible_state":[False,True],"early_return":[.1,.1],"late_return":[-.1,-.2],"completion_ratio":[.7,2.],"late_unanimous":[False,True]});a,s=support.active_and_side(f);assert a.tolist()==[False,True] and s.tolist()==[-1,-1];_,flip=support.active_and_side(f,"direction_flip");assert flip.tolist()==[1,1]

def test_controls_are_distinct():
 f=pd.DataFrame({"source_valid":[True,True],"eligible_state":[False,True],"early_return":[.1,.1],"late_return":[-.2,-.2],"completion_ratio":[2.,2.],"late_unanimous":[False,True]})
 late,_=support.active_and_side(f,"late_unanimity_only"); completion,_=support.active_and_side(f,"completion_dominance_only")
 assert late.tolist()==[False,True] and completion.tolist()==[True,False]

def test_outcomes_closed():
 source=open(support.__file__).read();assert '"execution_prices_opened": False' in source and '"gross9_rows_opened": False' in source and '"rv20_opened": False' in source
