import numpy as np
import pandas as pd
from training import build_high_volatility_regression_channel_breakout_relay_support as s

def test_lean_ols_endpoint_and_population_standard_deviation():
 close=pd.Series(3+2*np.arange(1.,21.));f=s.regression_channel(close,pd.Series([True]*20))
 assert np.isclose(f.loc[19,"intercept"],3) and np.isclose(f.loc[19,"slope"],2) and np.isclose(f.loc[19,"regression"],43)
 assert np.isclose(f.loc[19,"price_std"],np.std(close,ddof=0))
 assert np.isclose(f.loc[19,"upper_channel"],43+2*np.std(close,ddof=0))

def test_outer_breakout_emits_continuation_side():
 close=pd.Series([100.]*19+[200.]);f=s.regression_channel(close,pd.Series([True]*20))
 assert close.iloc[-1]>f.loc[19,"upper_channel"] and f.loc[19,"entry_side"]==1

def test_invalid_bar_resets_twenty_bar_window():
 close=pd.Series(np.arange(50,dtype=float)+100);valid=pd.Series([True]*21+[False]+[True]*28);f=s.regression_channel(close,valid)
 assert f.regression.first_valid_index()==19 and f.loc[22:].regression.first_valid_index()==41

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"center_side":[0,-1,0,1,0,-1],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_breakout");assert stale.iloc[3] and side.iloc[3]==1
 follow,side,_=s.active(panel(),"direction_flip");assert follow.iloc[2] and side.iloc[2]==-1
 center,side,_=s.active(panel(),"regression_center_crossover");assert center.iloc[1] and side.iloc[1]==-1
