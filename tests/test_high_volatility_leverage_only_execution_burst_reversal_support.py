import numpy as np
import pandas as pd
from training import build_high_volatility_leverage_only_execution_burst_reversal_support as s
def test_cash_unconfirmed_burst_fades_perpetual():
 side=s.burst_side(pd.Series([.01,-.01,.01]),pd.Series([0.,.002,.002]),pd.Series([True,True,True]));assert side.tolist()==[-1,1,0]
def test_primary_uses_frozen_fade_side():
 panel=pd.DataFrame({'source_valid':[True,True],'burst_side':[0,1],'execution_dominance':[0,2.],'dominance_rank':[.2,.999],'perpetual_return':[0,-.01],'spot_return':[0,.001],'variation_active':[True,True]});active,side,_=s.active(panel);assert active.tolist()==[False,True] and side.iloc[1]==1
def test_fast_midrank_is_strict_prior():
 values=pd.Series([1.,2.,2.,np.nan,3.,1.,4.]);expected=pd.Series([np.nan,np.nan,.75,np.nan,1.,0.,1.]);pd.testing.assert_series_equal(s.strict_prior_midrank(values,3,2),expected)
def test_pinned_registration():assert s.PREREG_SHA=='212ed0a7279bf4e7fa1293bacaeb7a006dac57b612f1e6ee63cbb542b98bcaca'
