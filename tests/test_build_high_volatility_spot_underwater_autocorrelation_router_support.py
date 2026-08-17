import numpy as np
import pandas as pd
from training import build_high_volatility_spot_underwater_autocorrelation_router_support as s
def states(values):
 d=pd.date_range("2023-01-01",periods=len(values),freq="8h",tz="UTC");return pd.DataFrame({"decision_time":d,"source_valid":True,"block_return":values})
def test_current_block_is_excluded_from_its_score():
 values=np.sin(np.arange(70));a=s.state_scores(states(values));values[-1]=1e6;b=s.state_scores(states(values));assert a.iloc[-1]["autocorrelation"]==b.iloc[-1]["autocorrelation"]
def test_negative_state_inverts_base_side():
 d=pd.Timestamp("2023-07-01T00:00Z");base=pd.DataFrame({"candidate":["b"],"control":["primary"],"split":["train"],"decision_time":[d],"feature_available_time":[d],"entry_time":[d+pd.Timedelta("5m")],"exit_time":[d+pd.Timedelta("8h5m")],"side":[1]});score=pd.DataFrame({"decision_time":[d],"history_count":[90],"autocorrelation":[-.2]});out=s.route(base,score);assert out.iloc[0]["side"]==-1
def test_real_run_is_deterministic(tmp_path):
 a=tmp_path/"s.csv.gz";b=tmp_path/"c.csv.gz";c=tmp_path/"r.json";first=s.run(a,b,c);raw=(a.read_bytes(),b.read_bytes(),c.read_bytes());second=s.run(a,b,c);assert raw==(a.read_bytes(),b.read_bytes(),c.read_bytes()) and first==second
