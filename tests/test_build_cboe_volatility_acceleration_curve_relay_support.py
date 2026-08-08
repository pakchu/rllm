import pandas as pd
from training import build_cboe_volatility_acceleration_curve_relay_support as support
def frame(dv,df,db):
 d=pd.to_datetime(["2024-06-03","2024-06-04","2024-06-05"]);return pd.DataFrame({"observation_date":d,"vix":[0.,dv,dv],"front":[0.,df,df],"broad":[0.,db,db],"delta_vix":[float("nan"),dv,0.],"delta_front":[float("nan"),df,0.],"delta_broad":[float("nan"),db,0.]})
def test_joint_acceleration_direction_and_causal_entry():
 lo=support.clock(frame(-.1,-.1,-.2));sh=support.clock(frame(.1,.1,.2));assert len(lo)==len(sh)==1 and lo.iloc[0].side==1 and sh.iloc[0].side==-1;assert lo.iloc[0].entry_time==pd.Timestamp("2024-06-05T13:35:00Z")
def test_three_way_disagreement_rejected_but_control_isolated():
 f=frame(-.1,.1,.1);assert support.clock(f).empty;assert len(support.clock(f,"no_vix_confirmation"))==1;assert len(support.clock(f,"vix_only"))==1
