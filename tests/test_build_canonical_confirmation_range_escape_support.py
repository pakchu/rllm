import numpy as np
import pandas as pd
from training import build_canonical_confirmation_range_escape_support as support

def _minutes(start,periods):
 t=pd.date_range(start,periods=periods,freq="1min");c=np.linspace(100,110,periods);o=np.r_[100.,c[:-1]];return pd.DataFrame({"ts":t,"open":o,"high":np.maximum(o,c)+.1,"low":np.minimum(o,c)-.1,"close":c,"duplicate_count":1})

def test_ceil():
 assert support.ceil_5m(301)==pd.Timestamp(600,unit="s",tz="UTC")

def test_strict_range_escape():
 start=pd.Timestamp("2024-01-01T00:00:00Z"); periods=289*5
 raw=_minutes(start,periods); raw.loc[raw.index[-1],["close","high"]]=[120.,120.1]
 market=support.prepare_minutes(raw); decision=start+pd.Timedelta(minutes=periods)
 value=support.range_escape(market,decision)
 assert value is not None and value[3]==1
 raw.loc[raw.index[-1],["close","high","low"]]=[109.,float(raw.loc[raw.index[-1],"open"])+.1,108.9]
 value=support.range_escape(support.prepare_minutes(raw),decision)
 assert value is not None and value[3]==0


def test_primary_flip_and_forced_long():
 f=pd.DataFrame({"source_valid":[True,True,True],"eligible_state":[0,-1,-1]})
 active,side=support.active_and_side(f); assert active.tolist()==[False,True,False] and side.tolist()==[0,-1,-1]
 _,flip=support.active_and_side(f,"direction_flip"); assert flip.tolist()==[0,1,1]
 _,forced=support.active_and_side(f,"same_clock_forced_long"); assert forced.tolist()==[0,1,1]

def test_outcomes_closed():
 source=open(support.__file__).read();assert '"execution_prices_opened": False' in source and '"gross9_rows_opened": False' in source and '"rv20_opened": False' in source
