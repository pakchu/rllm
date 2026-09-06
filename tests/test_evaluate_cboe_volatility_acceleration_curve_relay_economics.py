import math,hashlib,json
from pathlib import Path
import pandas as pd
from training import evaluate_cboe_volatility_acceleration_curve_relay_economics as economics

def test_exact_offset_funding_posts_to_containing_bar():
 start=pd.Timestamp("2023-07-01T00:00:00Z");end=start+pd.Timedelta(minutes=10);dates=pd.date_range(start,end,freq="5min",inclusive="both")
 market=pd.DataFrame({"date":dates,"open":100.0,"high":100.0,"low":100.0,"close":100.0});funding=pd.DataFrame({"date":[start+pd.Timedelta(milliseconds=5),end],"funding_rate":[.01,.5],"mark_price":[100.,100.]});clock=pd.DataFrame({"entry_time":[start],"exit_time":[end],"side":[1]})
 result=economics.engine.simulate(clock,market,funding,start,end,cost=0.0);assert math.isclose(result["final_equity"],.995,abs_tol=1e-12)

def test_frozen_accounting_and_stage_contract():
 assert economics.LEVERAGE==.5 and economics.BASE_COST==.0006 and economics.STRESS_COST==.001
 assert economics.STAGES["final"][2]=="2026-08-01T00:00:00Z"
 source=Path(economics.__file__).read_text();assert "full calendar including idle time" in source and "global peak, every held favorable then adverse" in source

def test_outcome_blind_evaluator_freeze_is_bound():
 assert hashlib.sha256(economics.FREEZE.read_bytes()).hexdigest()=="ed9d9ae6f6e91574e96906fc09a715b34ef9bbe8dcd3bcc6e6c0ca9193a9c9d3"
 d=json.loads(economics.FREEZE.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==economics.canonical_hash(core) and d['outcomes_opened'] is False
 assert d['evaluator']['sha256']==economics.sha256(Path(economics.__file__))
 novelty,freeze=economics.verify('train');assert novelty['advance_to_economic_outcomes'] is True and freeze['outcomes_opened'] is False
