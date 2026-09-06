import numpy as np,pandas as pd
from training import build_high_volatility_execution_count_elasticity_relay_support as s

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(range(181),dtype=float));assert np.isnan(r.iloc[179]);assert r.iloc[180]==1.
def test_elasticity_count_normalization():
 count,e=s.execution_elasticity(.04,np.full(480,4.));assert count==1920;assert e==.04/np.sqrt(1920)
 _,lower=s.execution_elasticity(.04,np.full(480,16.));assert lower<e
def test_invalid_counts_fail_closed():
 assert np.isnan(s.execution_elasticity(.01,np.full(480,-1.))[1]);assert np.isnan(s.execution_elasticity(.01,np.full(480,1.5))[1])
def test_onset_side_and_contract():
 x=pd.DataFrame({'source_valid':[True]*4,'block_return':[.01,.01,-.01,-.01],'variation_rank':[.6,.7,.8,.8],'elasticity_rank':[.8,.8,.8,.7],'execution_elasticity':[1.]*4,'return_rank':[.8]*4})
 a,side=s.active(x,'primary');assert a.tolist()==[False,True,False,False];assert side.tolist()==[1,1,-1,-1]
 assert s.PREREG_SHA=='62094724e9d4bcb58ec3f8582cb3ce0e296debbbbfc04bb04dae2332b5ad7d47'
