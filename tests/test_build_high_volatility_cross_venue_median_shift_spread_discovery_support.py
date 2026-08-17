import pandas as pd
from training import build_high_volatility_cross_venue_median_shift_spread_discovery_support as s
def panels(n=183):
 d=pd.date_range("2023-05-02T01:00Z",periods=n,freq="8h");perp=pd.DataFrame({"decision_time":d,"feature_available_time":d,"source_valid":True,"median_shift":0.,"variation_rank":.8});spot=pd.DataFrame({"decision_time":d,"source_valid":True,"spot_shift":[i/1000 for i in range(n)]});return perp,spot
def test_causal_rank_excludes_current_and_onset_is_strict():
 p,q=panels();x=s.prepare(p,q);assert x["spread_rank"].iloc[:180].isna().all();assert x["eligible"].sum()>=1
def test_clock_side_is_spread_sign():
 p,q=panels();x=s.prepare(p,q);clock=s.build_clock(x);assert len(clock)>=1 and set(clock["side"]).issubset({-1,1})
def test_real_run_is_deterministic(tmp_path):
 f=tmp_path/"f.csv.gz";c=tmp_path/"c.csv.gz";r=tmp_path/"r.json";first=s.run(f,c,r);raw=(f.read_bytes(),c.read_bytes(),r.read_bytes());second=s.run(f,c,r);assert raw==(f.read_bytes(),c.read_bytes(),r.read_bytes()) and first==second
