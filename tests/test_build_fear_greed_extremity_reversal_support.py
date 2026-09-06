import numpy as np,pandas as pd
from training import build_fear_greed_extremity_reversal_support as support
def test_fger_rank_is_strict_prior():
 r=support.strict_prior_midrank(pd.Series([1.,2.,3.]),lookback=2,minimum=2);assert np.isnan(r.iloc[:2]).all();assert r.iloc[2]==1.
def test_fger_primary_extremes_and_volatility_gate():
 f=pd.DataFrame({"fear_greed_value":[20,80,50,20],"value_classification":["Extreme Fear","Extreme Greed","Neutral","Extreme Fear"],"btc_variation_rank":[.9,.9,.9,.5]});assert support.signal(f,"primary").tolist()==[1,-1,0,0];assert support.signal(f,"direction_flip").tolist()==[-1,1,0,0]
def test_fger_clock_has_one_day_hold_and_half_open_reservation():
 f=pd.DataFrame({"sentiment_date":pd.to_datetime(["2024-06-30","2024-07-01"],utc=True),"decision_time":pd.to_datetime(["2024-07-01","2024-07-02"],utc=True),"fear_greed_value":[20,80],"value_classification":["Extreme Fear","Extreme Greed"],"btc_realized_variation":[.1,.1],"btc_variation_rank":[.9,.9]});c=support.build_clock(f);assert len(c)==2;assert pd.Timestamp(c.iloc[0].entry_time)==pd.Timestamp("2024-07-01T00:05Z");assert pd.Timestamp(c.iloc[0].exit_time)==pd.Timestamp("2024-07-02T00:05Z")
def test_fger_builder_is_outcome_blind():
 s=support.BUILDER.read_text();assert "funding_rates_binance" not in s;assert '"advance_to_economic_outcomes":False' in s;assert '"promotion_authorized":False' in s
