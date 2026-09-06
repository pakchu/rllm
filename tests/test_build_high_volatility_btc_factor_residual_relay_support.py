import numpy as np,pandas as pd
from training import build_high_volatility_btc_factor_residual_relay_support as support

def test_prior_ols_excludes_current_and_recovers_line():
 x=pd.Series(np.arange(181,dtype=float));y=1+2*x;r=support.causal_residuals(x,y);assert np.isnan(r.standardized_residual.iloc[:180]).all();assert abs(r.fit_alpha.iloc[180]-1)<1e-9;assert abs(r.fit_beta.iloc[180]-2)<1e-9
def test_rank_excludes_current():
 x=pd.Series(np.arange(181,dtype=float));r=support.prior_midrank(x);assert np.isnan(r.iloc[:180]).all();assert r.iloc[180]==1.
def frame():return pd.DataFrame({"source_valid":[True]*3,"standardized_residual":[1.2,-1.2,.5],"fixed_beta_one_z":[-1.,1.,-1.],"btc_return":[.1,-.1,.1],"variation_rank":[.7,.6,.7]})
def test_primary_and_controls():
 f=frame();a,s,_=support.conditions(f);assert a.tolist()==[True,False,False];assert s.tolist()==[1,-1,1];assert support.conditions(f,"no_variation_gate")[0].tolist()==[True,True,False];assert support.conditions(f,"no_residual_tail")[0].tolist()==[True,False,True];assert support.conditions(f,"raw_btc_return_side")[1].tolist()==[1,-1,1];assert support.conditions(f,"fixed_beta_one")[1].tolist()==[-1,1,-1]
def test_query_and_hash_bound():assert "bars_binance" in support.QUERY and support.PREREG_SHA=="de97c861daa3ea78241e6f996345b18f5fc8fb7869acfaa7e2550df02c254683"
