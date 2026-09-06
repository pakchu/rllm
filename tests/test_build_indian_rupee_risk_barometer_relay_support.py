import numpy as np
import pandas as pd
from training import build_indian_rupee_risk_barometer_relay_support as support

def test_irbr_rank_is_strict_prior():
 r=support.rank(pd.Series([1.,2.,3.]),lookback=2,minimum=2);assert np.isnan(r.iloc[:2]).all();assert r.iloc[2]==1.

def test_irbr_side_requires_fx_tail_and_volatility():
 f=pd.DataFrame({"usdinr_return":[.01,-.01,.01],"absolute_fx_return_rank":[.9,.9,.5],"btc_variation_rank":[.9,.5,.9]})
 assert support.signal(f,"primary").tolist()==[-1,0,0]
 assert support.signal(f,"no_fx_tail").tolist()==[-1,0,-1]
 assert support.signal(f,"direction_flip").tolist()==[1,0,0]

def test_irbr_clock_uses_1005_utc_and_twelve_hours():
 d=pd.Timestamp("2024-07-01 10:00",tz="UTC");f=pd.DataFrame({"session_date":[pd.Timestamp("2024-07-01")],"decision_time":[d],"usdinr_return":[.01],"absolute_fx_return_rank":[.9],"btc_realized_variation":[.1],"btc_variation_rank":[.9]});c=support.build_clock(f);assert len(c)==1;assert pd.Timestamp(c.iloc[0].entry_time)==d+pd.Timedelta(minutes=5);assert pd.Timestamp(c.iloc[0].exit_time)==d+pd.Timedelta(hours=12,minutes=5)

def test_irbr_builder_is_outcome_blind():
 s=support.BUILDER.read_text();assert "funding_rates_binance" not in s;assert '"advance_to_economic_outcomes":False' in s;assert '"promotion_authorized":False' in s
