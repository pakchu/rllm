import math
from pathlib import Path
import pandas as pd
from training import evaluate_dominant_quote_deleveraging_ignition_relay_economics as economics

def test_exact_offset_funding_posts_to_containing_bar():
 start=pd.Timestamp("2023-07-01T00:00:00Z");end=start+pd.Timedelta(minutes=10);dates=pd.date_range(start,end,freq="5min",inclusive="both")
 market=pd.DataFrame({"date":dates,"open":100.,"high":100.,"low":100.,"close":100.});funding=pd.DataFrame({"date":[start+pd.Timedelta(milliseconds=5),end],"funding_rate":[.01,.5],"mark_price":[100.,100.]});clock=pd.DataFrame({"entry_time":[start],"exit_time":[end],"side":[1]})
 result=economics.engine.simulate(clock,market,funding,start,end,cost=0.);assert math.isclose(result["final_equity"],.995,abs_tol=1e-12)

def test_strict_accounting_stage_and_controls_are_frozen():
 assert economics.LEVERAGE==.5 and economics.BASE_COST==.0006 and economics.STRESS_COST==.001 and economics.STAGES["final"][2]=="2026-08-01T00:00:00Z"
 source=Path(economics.__file__).read_text();assert "full calendar including idle time" in source and "global peak, every held favorable then adverse" in source
 assert economics.CONTROLS==("no_joint_expansion","no_oi_deleveraging","no_alternative_quiet","one_hour_stale_flow","direction_flip")
