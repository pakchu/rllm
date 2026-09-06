import pandas as pd
from training import build_daily_taker_flow_acceleration_relay_support as support
def frame():
 d=pd.Timestamp("2024-07-02T00:00:00Z");return pd.DataFrame({"source_day":[d-pd.Timedelta(days=1)],"decision_time":[d],"source_valid":[True],"normalized_flow":[.08],"flow_change":[.06],"flow_change_z":[1.5],"flow_level_z":[1.2],"realized_variation":[.06],"realized_variation_rank":[.8]})
def test_dtfar_follows_flow_acceleration_direction():
 c=support.clock(frame());assert len(c)==1;assert c.iloc[0].side==1;assert c.iloc[0].entry_time==pd.Timestamp("2024-07-02T00:05:00Z");assert c.iloc[0].exit_time-c.iloc[0].entry_time==pd.Timedelta(hours=12)
def test_dtfar_rejects_small_z_or_low_volatility():
 x=frame();x.loc[0,"flow_change_z"]=.9;assert support.clock(x).empty;x=frame();x.loc[0,"realized_variation_rank"]=.64;assert support.clock(x).empty
def test_dtfar_direction_flip_is_diagnostic_only():
 a=support.clock(frame());b=support.clock(frame(),"direction_flip");assert len(a)==len(b)==1;assert a.iloc[0].side==-b.iloc[0].side
def test_causal_statistics_exclude_current():
 v=pd.Series([float(i) for i in range(61)]);assert support.strict_prior_midrank(v).iloc[60]==1.;assert support.causal_z(v).iloc[:60].isna().all();assert support.causal_z(v).iloc[60]>1.
