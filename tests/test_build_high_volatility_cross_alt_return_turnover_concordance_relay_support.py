import numpy as np
import pandas as pd
from training import build_high_volatility_cross_alt_return_turnover_concordance_relay_support as s

def test_covariance_sign():
 returns=np.array([-.03,-.02,-.01,.01,.02,.03]);shocks=np.array([-3.,-2.,-1.,1.,2.,3.]);assert s.population_covariance(returns,shocks)>0 and s.population_covariance(returns,-shocks)<0

def test_prior_median_excludes_current():
 old=(s.P["turnover_history_decisions"],s.P["minimum_turnover_history_decisions"]);s.P["turnover_history_decisions"]=3;s.P["minimum_turnover_history_decisions"]=2
 try:
  x=s.prior_median(pd.Series([1.,3.,100.]));assert np.isnan(x.iloc[1]) and x.iloc[2]==2
 finally:s.P["turnover_history_decisions"],s.P["minimum_turnover_history_decisions"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(r.iloc[179]) and r.iloc[180]==1

def panel():return pd.DataFrame({"source_valid":[True]*5,"concordance":[-1.,1.,2.,-1.,1.],"raw_level_covariance":[-2.,1.,2.,-1.,1.],"variation_rank":[.8,.8,.4,.8,.8],"feature_available_time":pd.date_range("2024-01-01",periods=5,freq="8h",tz="UTC")})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[True,True,False,True,True] and z.tolist()==[-1,1,1,-1,1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[2]
 stale,side,_=s.active(panel(),"one_block_stale_concordance");assert not stale.iloc[0] and stale.iloc[1] and side.iloc[1]==-1
