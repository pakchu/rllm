import math
from pathlib import Path
import pandas as pd
from training import evaluate_high_volatility_intrinsic_topology_ridge_relay_economics as economics

def test_exact_offset_funding_posts_to_containing_bar():
 start=pd.Timestamp("2023-07-01T00:00:00Z");end=start+pd.Timedelta(minutes=10);dates=pd.date_range(start,end,freq="5min",inclusive="both");market=pd.DataFrame({"date":dates,"open":100.,"high":100.,"low":100.,"close":100.});funding=pd.DataFrame({"date":[start+pd.Timedelta(milliseconds=5),end],"funding_rate":[.01,.5],"mark_price":[100.,100.]});clock=pd.DataFrame({"entry_time":[start],"exit_time":[end],"side":[1]});assert math.isclose(economics.engine.simulate(clock,market,funding,start,end,cost=0.)["final_equity"],.995,abs_tol=1e-12)
def test_frozen_accounting_and_stage_contract():
 assert economics.POLICY_ID=="HVITR-8" and economics.LEVERAGE==.5 and economics.BASE_COST==.0006 and economics.STRESS_COST==.001;assert economics.STAGES["final"][2]=="2026-08-01T00:00:00Z";assert economics.CONTROLS==("no_volatility_gate","no_prediction_strength_gate","one_boundary_stale_features","direction_flip","forced_long");source=Path(economics.__file__).read_text();assert "full calendar including idle time" in source and "global peak, every held favorable then adverse" in source
