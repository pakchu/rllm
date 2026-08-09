import numpy as np,pandas as pd
from training import build_high_volatility_macro_polarity_concordance_relay_support as support
def test_rank_excludes_current():
 r=support.rank(pd.Series(np.arange(181,dtype=float)));assert r.iloc[:180].isna().all();assert r.iloc[180]==1.
def frame(**changes):
 row={"source_day":pd.Timestamp("2023-07-01T00:00:00Z"),"decision_time":pd.Timestamp("2023-07-09T00:00:00Z"),"state_valid":True,"news_change":.1,"epu_change":-.2,"polarity_concordance":True,"btc_variation":.2,"btc_variation_rank":.8};row.update(changes);return pd.DataFrame([row])
def test_primary_joint_polarity_and_controls():
 s=frame();assert support.build_clock(s).side.tolist()==[1];assert support.build_clock(s,"direction_flip").side.tolist()==[-1];assert support.build_clock(s,"same_clock_forced_long").side.tolist()==[1]
 bad=frame(epu_change=.2,polarity_concordance=False);assert support.build_clock(bad).empty;assert support.build_clock(bad,"news_only").side.tolist()==[1];assert support.build_clock(bad,"epu_only").side.tolist()==[-1]
def test_variation_gate_and_stale_pair_are_causal():
 low=frame(btc_variation_rank=.2);assert support.build_clock(low).empty;assert support.build_clock(low,"no_variation_gate").side.tolist()==[1]
 prior=frame(source_day=pd.Timestamp("2023-06-24T00:00:00Z"),decision_time=pd.Timestamp("2023-07-02T00:00:00Z"));current=frame(news_change=-.1,epu_change=.2);clock=support.build_clock(pd.concat([prior,current],ignore_index=True),"one_week_stale_pair");assert clock.source_day.tolist()==[pd.Timestamp("2023-06-24T00:00:00Z")];assert clock.side.tolist()==[1]
