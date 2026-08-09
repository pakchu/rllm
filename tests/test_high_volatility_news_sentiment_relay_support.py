import numpy as np,pandas as pd
from training import build_high_volatility_news_sentiment_relay_support as support
def test_rank_excludes_current():
 r=support.rank(pd.Series(np.arange(181,dtype=float)));assert r.iloc[:180].isna().all();assert r.iloc[180]==1.
def _states(**change):
 row={"source_day":pd.Timestamp("2023-07-01T00:00:00Z"),"decision_time":pd.Timestamp("2023-07-03T00:00:00Z"),"state_valid":True,"sentiment_change":.1,"sentiment_shock_rank":.8,"btc_variation":.1,"btc_variation_rank":.8};row.update(change);return pd.DataFrame([row])
def test_primary_follows_sentiment_change_and_controls_are_diagnostic():
 s=_states();c=support.build_clock(s);assert c.side.tolist()==[1];assert c.entry_time.iloc[0]==pd.Timestamp("2023-07-03T00:05:00Z");assert support.build_clock(s,"direction_flip").side.tolist()==[-1];assert support.build_clock(s,"same_clock_forced_long").side.tolist()==[1]
def test_sentiment_tail_is_required_only_by_control():
 s=_states(sentiment_shock_rank=.2);assert support.build_clock(s).empty;assert support.build_clock(s,"no_sentiment_tail").side.tolist()==[1]
