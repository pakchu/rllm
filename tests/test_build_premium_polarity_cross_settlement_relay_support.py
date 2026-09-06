import numpy as np
import pandas as pd
from training import build_premium_polarity_cross_settlement_relay_support as support

def _premium(start, closes):
 ts=pd.date_range(start,periods=len(closes),freq="1min"); values=np.asarray(closes,float)
 return pd.DataFrame({"ts":ts,"open":values,"high":values+.01,"low":values-.01,"close":values,"duplicate_count":1})


def test_polarity_cross_strict_60():
 start=pd.Timestamp("2024-01-01T00:00:00Z"); decision=start+pd.Timedelta(minutes=61)
 premium=support.prepare_premium(_premium(start,[.1]*60+[-.1]))
 value=support.polarity_cross(premium,decision)
 assert value is not None and value[2] is True and value[4]==-1
 mixed=[.1]*29+[-.1]+[.1]*30+[-.1]
 value=support.polarity_cross(support.prepare_premium(_premium(start,mixed)),decision)
 assert value is not None and value[2] is False


def test_primary_flip_and_forced_long():
 f=pd.DataFrame({"source_valid":[True,True],"eligible_state":[1,-1],"current_close":[.1,-.1],"persistent_30":[True,True]})
 active,side=support.active_and_side(f); assert active.tolist()==[True,True] and side.tolist()==[1,-1]
 _,flip=support.active_and_side(f,"direction_flip"); assert flip.tolist()==[-1,1]
 _,forced=support.active_and_side(f,"same_clock_forced_long"); assert forced.tolist()==[1,1]


def test_outcomes_closed():
 source=open(support.__file__).read();assert '"execution_prices_opened": False' in source and '"gross9_rows_opened": False' in source and '"rv20_opened": False' in source
