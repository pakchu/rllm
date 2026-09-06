import pandas as pd
from training import build_high_volatility_equity_duration_rotation_relay_support as s
def frame(spy=.01,tlt=-.01):
 d=pd.Timestamp("2023-07-03T23:00Z");return pd.DataFrame({"session_date":["2023-07-03"],"decision_time":[d],"source_valid":[True],"spy_return":[spy],"tlt_return":[tlt],"btc_variation":[.1],"btc_variation_rank":[.8]})
def test_opposite_session_returns_follow_spy():
 x=s.build_clock(frame());assert len(x)==1 and x.iloc[0]["side"]==1
def test_consensus_session_is_not_part_of_rotation():assert s.build_clock(frame(.01,.01)).empty
def test_real_run_is_deterministic(tmp_path):
 c=tmp_path/"c.csv.gz";r=tmp_path/"r.json";first=s.run(c,r);raw=(c.read_bytes(),r.read_bytes());second=s.run(c,r);assert raw==(c.read_bytes(),r.read_bytes()) and first==second
