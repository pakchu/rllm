import numpy as np,pandas as pd
from training import build_high_volatility_sign_run_asymmetry_relay_support as support

def test_run_lengths_use_strict_signs_and_zero_breaks_runs():
 assert support.run_lengths(np.array([1.,2.,-1.,-2.,-3.,0.,1.,2.,3.,4.]))==(4,3)
def test_rank_excludes_current():
 x=pd.Series(np.arange(253,dtype=float));r=support.prior_midrank(x);assert np.isnan(r.iloc[:252]).all();assert r.iloc[252]==1.
def frame():return pd.DataFrame({"source_valid":[True]*3,"run_asymmetry":[2.,-2.,2.],"block_return":[.1,.1,-.1],"run_share_rank":[.85,.85,.7],"variation_rank":[.7,.6,.7]})
def test_primary_and_controls():
 f=frame();a,s,_=support.conditions(f);assert a.tolist()==[True,False,False];assert s.tolist()==[1,-1,1];assert support.conditions(f,"no_variation_gate")[0].tolist()==[True,True,False];assert support.conditions(f,"no_run_share_gate")[0].tolist()==[True,False,True];assert support.conditions(f,"net_block_return_side")[1].tolist()==[1,1,-1];assert support.conditions(f,"direction_flip")[1].tolist()==[-1,1,-1]
def test_queries_and_prereg_hash():
 assert "bars_binance" in support.BAR_QUERY;assert "funding_rates_binance" not in support.BAR_QUERY;assert support.PREREG_SHA=="75dd3a7e286421d274335b139806bff0191b4bad499edf13f4f1e7b0963ed939"
