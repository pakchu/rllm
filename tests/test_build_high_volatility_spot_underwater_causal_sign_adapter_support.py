import pandas as pd
from training import build_high_volatility_spot_underwater_causal_sign_adapter_support as s
def labels(n=10):
 d=pd.date_range("2023-07-01",periods=n,freq="16h",tz="UTC");return pd.DataFrame({"candidate":"b","control":"primary","split":"train","decision_time":d,"feature_available_time":d,"entry_time":d+pd.Timedelta("5m"),"exit_time":d+pd.Timedelta("8h5m"),"side":1,"label_valid":True,"gross_directional_label":[.01]*n})
def test_adapter_uses_only_labels_mature_by_decision():
 x=labels();out=s.adapt(x);assert not out.empty and out["memory_latest_exit"].le(out["decision_time"]).all()
def test_negative_memory_inverts_base_side():
 x=labels();x["gross_directional_label"]=-.01;out=s.adapt(x);assert not out.empty and out.iloc[-1]["side"]==-1
def test_label_formula_uses_fixed_base_side():
 d=pd.Timestamp("2023-07-01T00:00Z");base=labels(1);opens=pd.DataFrame({"date":[base.iloc[0].entry_time,base.iloc[0].exit_time],"open":[100.,110.],"source_rows":[5,5]});x=s.label_base(base,opens);assert x.iloc[0]["gross_directional_label"]>0
