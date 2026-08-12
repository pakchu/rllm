import numpy as np
import pandas as pd
from training import build_high_volatility_leverage_displacement_amplification_reversal_support as s
def test_confirmed_leverage_amplification_fades_direction():
 assert s.amplification_side(pd.Series([.02,-.02,.02]),pd.Series([.01,-.01,-.01]),pd.Series([True]*3)).tolist()==[-1,1,0]
def test_primary_uses_fade_side():
 panel=pd.DataFrame({'source_valid':[True,True],'amplification_side':[0,-1],'leverage_excess':[0,.01],'excess_rank':[.2,.999],'cash_excess':[0,-.01],'cash_excess_rank':[.2,.1],'perpetual_return':[0,.02],'spot_return':[0,.01],'variation_active':[True,True]});active,side,_=s.active(panel);assert active.tolist()==[False,True] and side.iloc[1]==-1
def test_fast_midrank_is_strict_prior():
 values=pd.Series([1.,2.,2.,np.nan,3.,1.,4.]);expected=pd.Series([np.nan,np.nan,.75,np.nan,1.,0.,1.]);pd.testing.assert_series_equal(s.strict_prior_midrank(values,3,2),expected)
def test_pinned_registration():assert s.PREREG_SHA=='145d23f6a0ae48b1f7ab5af8ddc6f822292e2912b85ae43b24dc670cdbed830d'
