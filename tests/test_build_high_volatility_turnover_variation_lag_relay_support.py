import numpy as np
import pandas as pd
from training import build_high_volatility_turnover_variation_lag_relay_support as s
def test_timing_statistics_positive_lag():
 open_=np.repeat(100.,480);returns=np.zeros(480);returns[20]=.03;close=open_*np.exp(returns);quote=np.ones(480);quote[450]=1000
 block=pd.DataFrame({"open":open_,"high":np.maximum(open_,close),"low":np.minimum(open_,close),"close":close,"quote_asset_volume":quote})
 lag,backloading,var,ret,final_ret=s.timing_statistics(block);assert lag>0 and backloading>0 and var>0 and np.isfinite(ret) and np.isfinite(final_ret)
def panel():
 return pd.DataFrame({"source_valid":[True]*6,"timing_lag":[.1,.3,.4,.2,.5,.4],"lag_rank":[.5,.8,.9,.4,.8,.9],"fixed_half_backloading":[.1,.3,.4,.2,.5,.4],"backloading_rank":[.5,.8,.9,.4,.8,.9],"variation_rank":[.8]*6,"completed_return":[.01,.02,.03,-.01,-.02,.01],"final_two_hour_return":[.01,.01,.02,-.01,-.01,.01],"feature_available_time":pd.date_range("2024-01-01",periods=6,freq="8h",tz="UTC")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,False,False,True,False] and side[active].tolist()==[1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 x=panel();x.loc[1,"lag_rank"]=.4;assert s.active(x,"fixed_half_backloading")[0].iloc[1]
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
