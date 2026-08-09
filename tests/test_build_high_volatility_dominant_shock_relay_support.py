import numpy as np,pandas as pd
from training import build_high_volatility_dominant_shock_relay_support as support

def test_rank_excludes_current():
 x=pd.Series(np.arange(181,dtype=float));r=support.prior_midrank(x);assert np.isnan(r.iloc[:180]).all();assert r.iloc[180]==1.
def frame():return pd.DataFrame({"source_valid":[True]*3,"dominant_return":[.1,-.1,.1],"latest_dominant_return":[.1,.1,-.1],"day_return":[-.1,-.1,.1],"dominant_share_rank":[.7,.7,.6],"variation_rank":[.7,.6,.7]})
def test_primary_and_controls():
 f=frame();a,s,_=support.conditions(f);assert a.tolist()==[True,False,False];assert s.tolist()==[1,-1,1];assert support.conditions(f,"no_variation_gate")[0].tolist()==[True,True,False];assert support.conditions(f,"no_dominant_share_gate")[0].tolist()==[True,False,True];assert support.conditions(f,"completed_day_return_side")[1].tolist()==[-1,-1,1];assert support.conditions(f,"latest_maximum_tie_break")[1].tolist()==[1,1,-1]
def test_query_and_hash_bound():assert "bars_binance" in support.QUERY and support.PREREG_SHA=="3f821d75a17044be86c1f4430e324bc209a19e9204e6553dd27e036ef0bd8466"
