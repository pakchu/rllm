import numpy as np,pandas as pd
import json
from training import build_yen_led_intraday_risk_transmission_relay_support as support
from training import preregister_yen_led_intraday_risk_transmission_relay as prereg
def test_ylirtr_rank_is_strict_prior():
 r=support.rank(pd.Series([1.,2.,3.]),lookback=2,minimum=2);assert np.isnan(r.iloc[:2]).all();assert r.iloc[2]==1.
def test_ylirtr_side_and_volatility_gate():
 f=pd.DataFrame({"usdjpy_return":[.01,-.01,.01],"lead_correlation":[.1,.1,.1],"lead_correlation_rank":[.9,.9,.9],"btc_variation_rank":[.9,.9,.5]});assert support.signal(f,"primary").tolist()==[1,-1,0];assert support.signal(f,"direction_flip").tolist()==[-1,1,0]
def test_ylirtr_clock_uses_2105_utc_and_twelve_hours():
 d=pd.Timestamp("2024-07-01 21:00",tz="UTC");f=pd.DataFrame({"session_date":[pd.Timestamp("2024-07-01")],"decision_time":[d],"usdjpy_return":[.01],"lead_correlation":[.1],"lead_correlation_rank":[.9],"btc_realized_variation":[.1],"btc_variation_rank":[.9]});c=support.build_clock(f);assert len(c)==1;assert pd.Timestamp(c.iloc[0].entry_time)==d+pd.Timedelta(minutes=5);assert pd.Timestamp(c.iloc[0].exit_time)==d+pd.Timedelta(hours=12,minutes=5)
def test_ylirtr_builder_is_outcome_blind():
 s=support.BUILDER.read_text();assert "funding_rates_binance" not in s;assert '"advance_to_economic_outcomes":False' in s;assert '"promotion_authorized":False' in s
def test_ylirtr_frozen_preregistration_matches_builder():
 assert json.loads(prereg.DEFAULT_OUTPUT.read_text())==prereg.build()
