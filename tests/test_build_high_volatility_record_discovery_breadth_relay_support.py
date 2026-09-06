import numpy as np,pandas as pd
from training import build_high_volatility_record_discovery_breadth_relay_support as support

def test_discovery_counts_excludes_first_bar_and_counts_strict_records():
 assert support.discovery_counts(np.array([2.,3.,3.,4.,1.]),np.array([2.,2.,1.,1.,0.]))==(2,2)
def test_rank_excludes_current():
 x=pd.Series(np.arange(253,dtype=float));r=support.prior_midrank(x);assert np.isnan(r.iloc[:252]).all();assert r.iloc[252]==1.
def frame():return pd.DataFrame({"source_valid":[True]*3,"breadth":[2.,-2.,2.],"block_return":[.1,.1,-.1],"discovery_share_rank":[.7,.7,.6],"variation_rank":[.7,.6,.7]})
def test_primary_and_controls():
 f=frame();a,s,_=support.conditions(f);assert a.tolist()==[True,False,False];assert s.tolist()==[1,-1,1];assert support.conditions(f,"no_variation_gate")[0].tolist()==[True,True,False];assert support.conditions(f,"no_discovery_share_gate")[0].tolist()==[True,False,True];assert support.conditions(f,"net_block_return_side")[1].tolist()==[1,1,-1];assert support.conditions(f,"direction_flip")[1].tolist()==[-1,1,-1]
def test_queries_and_prereg_hash():
 assert "bars_binance" in support.BAR_QUERY;assert "funding_rates_binance" not in support.BAR_QUERY;assert support.PREREG_SHA=="4f7e4f9d00ea0951325faf48ae73c738a578db4fc92a527104ca55b589ce070a"
