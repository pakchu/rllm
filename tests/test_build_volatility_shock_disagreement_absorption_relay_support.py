import pandas as pd
from training import build_volatility_shock_disagreement_absorption_relay_support as s
def frame(dv,dr,rank=.8):
 d=pd.to_datetime(["2024-06-03","2024-06-04","2024-06-05"]);return pd.DataFrame({"observation_date":d,"delta_log_vix":[float("nan"),dv,0.],"delta_relative_convexity":[float("nan"),dr,0.],"absolute_vix_change_rank":[float("nan"),rank,.5]})
def test_vsdar_absorbs_unconfirmed_vix_shock_next_session():
 lo=s.clock(frame(.1,-.1));sh=s.clock(frame(-.1,.1));assert len(lo)==len(sh)==1 and lo.iloc[0].side==1 and sh.iloc[0].side==-1;assert lo.iloc[0].entry_time==pd.Timestamp("2024-06-05T13:35:00Z")
def test_vsdar_rejects_small_or_confirmed_shock():
 assert s.clock(frame(.1,-.1,.74)).empty;assert s.clock(frame(.1,.1)).empty;assert len(s.clock(frame(.1,-.1,.74),"no_magnitude_gate"))==1
