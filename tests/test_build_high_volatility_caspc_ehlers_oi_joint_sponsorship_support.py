import pandas as pd
from training import build_high_volatility_caspc_ehlers_oi_joint_sponsorship_support as s
def row(candidate,when,side):
 t=pd.Timestamp(when);return {"candidate":candidate,"control":"primary","split":"train","decision_time":t,"feature_available_time":t,"entry_time":t+pd.Timedelta(minutes=5),"exit_time":t+pd.Timedelta(hours=8,minutes=5),"side":side}
def test_intersection_requires_exact_time_and_side():
 left=pd.DataFrame([row('a','2023-07-01T03:00Z',1),row('a','2023-07-02T03:00Z',-1)])
 right=pd.DataFrame([row('b','2023-07-01T03:00Z',1),row('b','2023-07-02T03:00Z',1)])
 out=s.intersect(left,right);assert len(out)==1 and out.iloc[0].side==1
def test_frozen_bindings():
 assert s.PREREG_SHA=="0a17e358f1b0838e5cc4a90556c6e328e98f73279631fa7c0030af071ac523fa"
 assert s.MINIMUM_EVENTS=={"train":8,"test":12,"eval":12,"final":8}
