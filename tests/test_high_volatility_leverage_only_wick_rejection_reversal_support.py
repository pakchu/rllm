import numpy as np
import pandas as pd
from training import build_high_volatility_leverage_only_wick_rejection_reversal_support as s
def test_exactly_one_excursion_defines_fade_side():
 assert s.directional_rejection(pd.Series([True,False,True]),pd.Series([False,True,True])).tolist()==[-1,1,0]
def test_primary_fades_leverage_only_wick():
 panel=pd.DataFrame({'source_valid':[True,True],'rejection_side':[0,-1],'upper_excess':[0,.1],'lower_excess':[0,0],'upper_excess_rank':[.2,.999],'lower_excess_rank':[.2,.1],'perpetual_return':[0,-.01],'spot_return':[0,0],'variation_active':[True,True]});active,side,_=s.active(panel);assert active.tolist()==[False,True] and side.iloc[1]==-1
def test_fast_midrank_is_strict_prior():
 values=pd.Series([1.,2.,2.,np.nan,3.,1.,4.]);expected=pd.Series([np.nan,np.nan,.75,np.nan,1.,0.,1.]);pd.testing.assert_series_equal(s.strict_prior_midrank(values,3,2),expected)
def test_pinned_registration():assert s.PREREG_SHA=='fe53b114eef4515f29aa4f34a07bea3cfc48c8127640edda34daa4708aa88470'
