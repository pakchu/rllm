import numpy as np,pandas as pd
from training import build_premium_price_latency_assimilation_relay_support as support
def test_prior_stats_are_strict():
 x=pd.Series(range(361),dtype=float);r=support.prior_midrank(x);m=support.prior_median(x);assert np.isnan(r.iloc[359]) and r.iloc[360]==1.;assert m.iloc[360]==179.5
def test_primary_onset_and_side():
 f=pd.DataFrame({"source_valid":[1,1,1],"stale_source_valid":[1,1,1],"early_premium_change":[1,1,1],"late_premium_change":[1,1,1],"early_btc_return":[0,0,0],"late_btc_return":[.001,.001,.001],"stale_early_premium_change":[1,1,1],"stale_late_premium_change":[1,1,1],"early_premium_rank":[.7,.9,.9],"early_btc_rank":[.2,.2,.2],"late_btc_abs_median":[.01,.01,.01],"late_btc_rank":[.2,.2,.2],"late_premium_rank":[.2,.2,.2],"stale_early_premium_rank":[.7,.9,.9]});a,s=support.active_and_side(f);assert a.tolist()==[False,True,False] and s.tolist()==[1,1,1];_,z=support.active_and_side(f,"direction_flip");assert z.tolist()==[-1,-1,-1]
def test_outcomes_closed():
 s=open(support.__file__).read();assert '"execution_prices_opened":False' in s and '"gross9_rows_opened":False' in s and '"rv20_opened":False' in s
