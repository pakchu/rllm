import numpy as np,pandas as pd
from training import build_high_volatility_extreme_funding_residual_relay_support as support
def test_prior_median_and_rank_exclude_current():
 v=pd.Series(np.arange(181,dtype=float));m=support.prior_median(v);r=support.rank(v)
 assert m.iloc[:60].isna().all() and m.iloc[60]==29.5
 assert r.iloc[:180].isna().all() and r.iloc[180]==1.
def test_extreme_residual_maps_contrarian_side():
 d=pd.to_datetime(["2023-07-01T00:00:00Z","2023-07-01T08:00:00Z"]);x=pd.DataFrame({"decision_time":d,"funding_rate":[.001,-.001],"funding_median":[0.,0.],"funding_residual":[.001,-.001],"residual_rank":[.8,.8],"btc_variation":[1.,1.],"variation_rank":[.8,.8],"source_valid":[True,True]});c=support.clock(x)
 assert c.side.tolist()==[-1,1] and (c.exit_time-c.entry_time).eq(pd.Timedelta(hours=8)).all()
