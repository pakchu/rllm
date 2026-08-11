import hashlib
import numpy as np
import pandas as pd
from training import build_high_volatility_new_york_opening_momentum_relay_support as s

def test_weekday_anchors_follow_new_york_dst():
 anchors=s.weekday_anchors(pd.Timestamp("2024-03-08T00:00:00Z"),pd.Timestamp("2024-03-13T00:00:00Z"))
 assert anchors.tolist()==[pd.Timestamp("2024-03-08T14:30:00Z"),pd.Timestamp("2024-03-11T13:30:00Z"),pd.Timestamp("2024-03-12T13:30:00Z")]

def test_rank_is_strict_prior():
 dates=pd.date_range("2022-12-01","2023-09-01",freq="5min",tz="UTC",inclusive="left");price=np.exp(np.arange(len(dates))*1e-7)*100
 market=pd.DataFrame({"date":dates,"open":price,"high":price*1.001,"low":price*.999,"close":price*1.00001});states=s.score_states(market)
 assert states.variation_rank.iloc[:120].isna().all() and np.isfinite(states.variation_rank.iloc[120])

def states():
 return pd.DataFrame([{"anchor_time":pd.Timestamp("2023-07-06T13:30:00Z"),"decision_time":pd.Timestamp("2023-07-06T14:00:00Z"),"reaction_return":.01,"first_fifteen_return":-.01,"pre_anchor_variation":.1,"variation_rank":.8}])

def test_primary_continues_and_controls_are_frozen():
 primary=s.build_clock(states());first=s.build_clock(states(),"first_fifteen_minute_direction");flip=s.build_clock(states(),"direction_flip");forced=s.build_clock(states(),"forced_long")
 assert primary.side.tolist()==[1] and first.side.tolist()==[-1] and flip.side.tolist()==[-1] and forced.side.tolist()==[1]
 assert primary.entry_time.tolist()==[pd.Timestamp("2023-07-06T14:05:00Z")] and primary.exit_time.tolist()==[pd.Timestamp("2023-07-06T20:00:00Z")]

def test_blind_and_bound():
 assert s.sha(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA and "funding" not in s.QUERY.lower() and "gross9" not in s.QUERY.lower()
