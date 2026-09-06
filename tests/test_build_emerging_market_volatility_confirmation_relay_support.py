import numpy as np
import pandas as pd
from training import build_emerging_market_volatility_confirmation_relay_support as support

def test_emvcr_causal_statistics_exclude_current():
 z=support.causal_z(pd.Series([1.,2.,3.,100.]),lookback=3,minimum=3);assert np.isnan(z.iloc[:3]).all();assert z.iloc[3]==98.
 r=support.strict_prior_midrank(pd.Series([1.,2.,3.]),lookback=2,minimum=2);assert np.isnan(r.iloc[:2]).all();assert r.iloc[2]==1.
def test_emvcr_primary_requires_confirmation_and_high_volatility():
 f=pd.DataFrame({"relative_volatility_shock":[.1,.1,.1],"shock_z":[1.,1.,1.],"btc_overnight_return":[-.01,.01,-.01],"btc_variation_rank":[.9,.9,.5],"vxeem_change":[.1,.1,.1],"vxeem_change_z":[1.,1.,1.]})
 assert support.signal(f,"primary").tolist()==[-1,0,0]
 assert support.signal(f,"no_btc_confirmation").tolist()==[-1,-1,0]
def test_emvcr_clock_is_0935_new_york_with_eight_hour_hold():
 d=pd.Timestamp("2024-07-01 09:30",tz=support.NY).tz_convert("UTC");f=pd.DataFrame({"cboe_observation_date":[pd.Timestamp("2024-06-28")],"next_cboe_source_date":[pd.Timestamp("2024-07-01")],"decision_time":[d],"relative_volatility_shock":[.1],"shock_z":[1.],"vxeem_change":[.1],"vxeem_change_z":[1.],"btc_overnight_return":[-.01],"btc_realized_variation":[.1],"btc_variation_rank":[.9]});c=support.build_clock(f);assert len(c)==1;assert pd.Timestamp(c.iloc[0].entry_time)==d+pd.Timedelta(minutes=5);assert pd.Timestamp(c.iloc[0].exit_time)==d+pd.Timedelta(hours=8,minutes=5)
def test_emvcr_builder_is_outcome_blind():
 s=support.BUILDER.read_text();assert "funding_rates_binance" not in s;assert '"advance_to_economic_outcomes": False' in s;assert '"promotion_authorized": False' in s
