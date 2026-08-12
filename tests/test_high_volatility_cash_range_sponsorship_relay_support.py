import numpy as np
import pandas as pd
from training import build_high_volatility_cash_range_sponsorship_relay_support as s
def test_confirmed_cash_range_follows_common_direction():
 assert s.confirmed_side(pd.Series([.01,-.01,.01]),pd.Series([.02,-.02,-.02]),pd.Series([True]*3)).tolist()==[1,-1,0]
def test_primary_uses_confirmed_side():
 panel=pd.DataFrame({'source_valid':[True,True],'range_side':[0,1],'cash_range_excess':[0,.1],'range_rank':[.2,.999],'perpetual_range_excess':[0,-.1],'perpetual_range_rank':[.2,.1],'perpetual_return':[0,.01],'spot_return':[0,.02],'variation_active':[True,True]});active,side,_=s.active(panel);assert active.tolist()==[False,True] and side.iloc[1]==1
def test_fast_midrank_is_strict_prior():
 values=pd.Series([1.,2.,2.,np.nan,3.,1.,4.]);expected=pd.Series([np.nan,np.nan,.75,np.nan,1.,0.,1.]);pd.testing.assert_series_equal(s.strict_prior_midrank(values,3,2),expected)
def test_pinned_registration():assert s.PREREG_SHA=='40f49daffc1346d51c86f8f51aa2f5391d9e3f5c035883920c49594903414eaa'
