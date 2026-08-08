import math
from pathlib import Path
import pandas as pd
from training import evaluate_cboe_front_slope_crypto_volatility_transmission_relay_economics as economics
def test_exact_offset_funding_posts_to_containing_bar():
 s=pd.Timestamp("2023-07-01T00:00:00Z");e=s+pd.Timedelta(minutes=10);m=pd.DataFrame({"date":pd.date_range(s,e,freq="5min",inclusive="both"),"open":100.,"high":100.,"low":100.,"close":100.});f=pd.DataFrame({"date":[s+pd.Timedelta(milliseconds=5),e],"funding_rate":[.01,.5],"mark_price":[100.,100.]});c=pd.DataFrame({"entry_time":[s],"exit_time":[e],"side":[1]});assert math.isclose(economics.engine.simulate(c,m,f,s,e,cost=0.)["final_equity"],.995,abs_tol=1e-12)
def test_strict_accounting_and_controls_are_frozen():
 assert economics.LEVERAGE==.5 and economics.BASE_COST==.0006 and economics.STRESS_COST==.001;source=Path(economics.__file__).read_text();assert "full calendar including idle time" in source and "global peak, every held favorable then adverse" in source;assert economics.CONTROLS==("no_crypto_confirmation","bvol_only_confirmation","dvol_only_confirmation","one_session_stale_front_slope_change","direction_flip")
