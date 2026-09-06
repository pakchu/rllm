import pandas as pd
from training import build_high_volatility_risk_parity_consensus_transition_support as s
def frame(values=(-.01,.01),tlt=None):
 d=pd.date_range("2023-07-03T23:00Z",periods=2,freq="24h");tlt=values if tlt is None else tlt;return pd.DataFrame({"session_date":["2023-07-03","2023-07-04"],"decision_time":d,"source_valid":[True,True],"spy_return":values,"tlt_return":tlt,"btc_variation":[.1,.1],"btc_variation_rank":[.8,.8]})
def test_opposite_consensus_signs_follow_new_common_side():
 x=s.build_clock(frame());assert len(x)==1 and x.iloc[0]["side"]==1
def test_nonconsensus_session_breaks_transition():assert s.build_clock(frame((-.01,.01),(.01,.01))).empty
def test_real_run_is_deterministic(tmp_path):
 c=tmp_path/"c.csv.gz";r=tmp_path/"r.json";first=s.run(c,r);raw=(c.read_bytes(),r.read_bytes());second=s.run(c,r);assert raw==(c.read_bytes(),r.read_bytes()) and first==second
