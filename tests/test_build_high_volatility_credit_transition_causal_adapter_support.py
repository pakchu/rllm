import pandas as pd
from training import build_high_volatility_credit_transition_causal_adapter_support as s
def events(n=10):
 d=pd.date_range("2023-07-01T23:00Z",periods=n,freq="48h");return pd.DataFrame({"session_date":d.date,"decision_time":d,"entry_time":d+pd.Timedelta("5m"),"exit_time":d+pd.Timedelta("24h5m"),"transition_side":1,"label_valid":True,"gross_directional_label":.01,"btc_variation_rank":.8})
def test_only_mature_labels_are_used():
 out=s.adapt(events());assert not out.empty and out["memory_latest_exit"].le(out["decision_time"]).all()
def test_negative_memory_inverts_transition():
 x=events();x["gross_directional_label"]=-.01;assert s.adapt(x).iloc[-1]["side"]==-1
def test_transition_requires_polarity_change():
 d=pd.date_range("2023-01-01T23:00Z",periods=3,freq="24h");x=pd.DataFrame({"session_date":d.date,"decision_time":d,"source_valid":True,"relative_credit_return":[.01,.02,-.01],"btc_variation_rank":.8});assert len(s.transitions(x))==1
