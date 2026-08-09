import numpy as np
import pandas as pd
from training import build_confirmation_ladder_directional_coherence_relay_support as support

def _minutes(start, periods, rising=True):
    times=pd.date_range(start,periods=periods,freq="1min");close=np.linspace(100,110,periods) if rising else np.linspace(100,90,periods);open_=np.r_[100.,close[:-1]]
    return pd.DataFrame({"ts":times,"open":open_,"high":np.maximum(open_,close)+.1,"low":np.minimum(open_,close)-.1,"close":close,"duplicate_count":1})

def test_rank_and_ceil():
    r=support.strict_prior_midrank(pd.Series(range(113),dtype=float));assert np.isnan(r.iloc[111]) and r.iloc[112]==1.;assert support.ceil_5m(301)==pd.Timestamp(600,unit="s",tz="UTC")

def test_interval_return_uses_complete_minutes():
    start=pd.Timestamp("2024-01-01T00:00:00Z");market=support.prepare_minutes(_minutes(start,10));value=support.interval_return(market,int(start.timestamp()),int((start+pd.Timedelta(minutes=10)).timestamp()));assert value is not None and value[0]>0 and value[1]==10

def test_invalid_duration_rejected():
    start=pd.Timestamp("2024-01-01T00:00:00Z");market=support.prepare_minutes(_minutes(start,40));assert support.interval_return(market,int(start.timestamp()),int((start+pd.Timedelta(minutes=31)).timestamp())) is None

def test_onset_preserves_state_across_invalid_anchor():
    f=pd.DataFrame({"source_valid":[True,False,True,True],"eligible_state":[True,False,True,False],"cumulative_return":[.1,np.nan,.2,-.1],"cumulative_abs_rank":[.95,np.nan,.95,.95],"path_agree_count":[5,0,5,5]});a,s=support.active_and_side(f);assert a.tolist()==[True,False,False,False] and s.tolist()==[1,0,1,-1]

def test_outcomes_closed():
    source=open(support.__file__).read();assert '"execution_prices_opened": False' in source and '"gross9_rows_opened": False' in source and '"rv20_opened": False' in source
