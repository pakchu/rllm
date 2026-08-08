import pandas as pd
from training import build_daily_variance_backloading_continuation_relay_support as support
def frame():
 d=pd.Timestamp("2024-07-02T00:00:00Z");return pd.DataFrame({"decision_time":[d],"source_valid_day":[True],"daily_realized_variation":[.06],"realized_variation_rank":[.8],"final_12h_variance_share":[.7],"final_12h_return":[.04],"final_6h_return":[.03],"first_6h_return":[-.01]})
def test_dvbcr_follows_late_impulse_direction():
 c=support.clock(frame());assert len(c)==1;assert c.iloc[0].side==1;assert c.iloc[0].entry_time==pd.Timestamp("2024-07-02T00:05:00Z");assert c.iloc[0].exit_time-c.iloc[0].entry_time==pd.Timedelta(hours=12)
def test_dvbcr_rejects_low_vol_backloading_or_dominance():
 x=frame();x.loc[0,"realized_variation_rank"]=.64;assert support.clock(x).empty;x=frame();x.loc[0,"final_12h_variance_share"]=.64;assert support.clock(x).empty;x=frame();x.loc[0,"final_6h_return"]=.019;assert support.clock(x).empty
def test_dvbcr_controls_are_diagnostic_only():
 a=support.clock(frame());b=support.clock(frame(),"direction_flip");e=support.clock(frame(),"first_six_hour_direction");assert len(a)==len(b)==len(e)==1;assert a.iloc[0].side==-b.iloc[0].side;assert e.iloc[0].side==-1
def test_midrank_excludes_current():
 r=support.strict_prior_midrank(pd.Series([float(i) for i in range(127)]));assert r.iloc[:126].isna().all();assert r.iloc[126]==1.
