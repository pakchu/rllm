import pandas as pd
from training import build_cboe_volatility_curve_impulse_relay_support as support
def frame(df,db,v=.8):
 d=pd.to_datetime(["2024-06-03","2024-06-04","2024-06-05","2024-06-06"]);return pd.DataFrame({"observation_date":d,"front":[0.,df,df,df],"broad":[0.,db,db,db],"delta_front":[float("nan"),df,0.,0.],"delta_broad":[float("nan"),db,0.,0.],"vix_rank":[v]*4})
def test_joint_curve_impulse_direction_and_causal_entry():
 lo=support.clock(frame(-.1,-.2));sh=support.clock(frame(.1,.2));assert len(lo)==len(sh)==1 and lo.iloc[0].side==1 and sh.iloc[0].side==-1;assert lo.iloc[0].entry_time==pd.Timestamp("2024-06-05T13:35:00Z")
def test_disagreement_and_low_vix_rejected():
 assert support.clock(frame(.1,-.1)).empty;assert support.clock(frame(.1,.1,.59)).empty;assert len(support.clock(frame(.1,.1,.59),"no_vix_high"))==1
def test_repeated_same_state_requires_onset():
 f=frame(.1,.1);f.loc[2,["delta_front","delta_broad"]]=[.1,.1];assert len(support.clock(f))==1;assert len(support.clock(f,"no_onset"))==2
