import pandas as pd
from training import build_daily_aggressive_flow_saturation_reversal_support as support
def frame():
 d=pd.Timestamp("2024-07-02T00:00:00Z");return pd.DataFrame({"source_day":[d-pd.Timedelta(days=1)],"decision_time":[d],"source_valid":[True],"normalized_flow":[-.08],"price_return":[-.04],"realized_variation":[.06],"realized_variation_rank":[.8]})
def test_dafsr_reverses_same_direction_flow_saturation():
 c=support.clock(frame());assert len(c)==1;assert c.iloc[0].side==1;assert c.iloc[0].entry_time==pd.Timestamp("2024-07-02T00:05:00Z");assert c.iloc[0].exit_time-c.iloc[0].entry_time==pd.Timedelta(hours=12)
def test_dafsr_rejects_disagreement_small_flow_or_low_vol():
 x=frame();x.loc[0,"normalized_flow"]=.08;assert support.clock(x).empty;x=frame();x.loc[0,"normalized_flow"]=-.04;assert support.clock(x).empty;x=frame();x.loc[0,"realized_variation_rank"]=.64;assert support.clock(x).empty
def test_dafsr_price_continuation_is_diagnostic_only():
 a=support.clock(frame());b=support.clock(frame(),"price_continuation");assert len(a)==len(b)==1;assert a.iloc[0].side==-b.iloc[0].side
def test_midrank_excludes_current():
 r=support.strict_prior_midrank(pd.Series([float(i) for i in range(61)]));assert r.iloc[:60].isna().all();assert r.iloc[60]==1.
