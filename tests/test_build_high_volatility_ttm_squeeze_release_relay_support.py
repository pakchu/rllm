import numpy as np
import pandas as pd
from training import build_high_volatility_ttm_squeeze_release_relay_support as s

def test_regression_last_returns_newest_fitted_value():
 series=pd.Series(np.arange(5,dtype=float));result=s.regression_last(series,pd.Series([True]*5),3);assert np.isclose(result.iloc[5],4)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"squeeze_on":[False,True,False,False,True,False],"side":[-1,-1,1,1,-1,-1],"bar_close":[2.]*6,"close_sma":[1.]*6,"variation_rank":[.8,.8,.8,.8,.8,.4]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[5]
 stale,side,_=s.active(panel(),"one_bar_stale_release");assert stale.iloc[3] and side.iloc[3]==1
