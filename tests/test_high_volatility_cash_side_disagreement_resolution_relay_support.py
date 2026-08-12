import numpy as np
import pandas as pd
from training import build_high_volatility_cash_side_disagreement_resolution_relay_support as s
def test_opposite_sign_tail_follows_spot():
 assert s.disagreement_side(pd.Series([.01,-.01,.01]),pd.Series([-.02,.02,.02]),pd.Series([True]*3)).tolist()==[-1,1,0]
def test_primary_uses_cash_side():
 panel=pd.DataFrame({'source_valid':[True,True],'disagreement_side':[0,-1],'absolute_divergence':[0,.03],'divergence_rank':[.2,.999],'perpetual_return':[0,.01],'spot_return':[0,-.02],'variation_active':[True,True]});active,side,_=s.active(panel);assert active.tolist()==[False,True] and side.iloc[1]==-1
def test_fast_midrank_is_strict_prior():
 values=pd.Series([1.,2.,2.,np.nan,3.,1.,4.]);expected=pd.Series([np.nan,np.nan,.75,np.nan,1.,0.,1.]);pd.testing.assert_series_equal(s.strict_prior_midrank(values,3,2),expected)
def test_pinned_registration():assert s.PREREG_SHA=='7ebe356ee73fe0036c57a3a9e6b14dd0486af01108d2da05bfec53f5c91b8fd0'
