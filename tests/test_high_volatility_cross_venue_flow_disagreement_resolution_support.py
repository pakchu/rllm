import numpy as np
import pandas as pd
from training import build_high_volatility_cross_venue_flow_disagreement_resolution_support as s
def test_opposite_flow_tail_follows_spot():
 assert s.flow_side(pd.Series([.2,-.2,.2]),pd.Series([-.3,.3,.3]),pd.Series([True]*3)).tolist()==[-1,1,0]
def test_primary_uses_spot_flow_side():
 panel=pd.DataFrame({'source_valid':[True,True],'flow_side':[0,-1],'flow_divergence':[0,.5],'divergence_rank':[.2,.999],'perpetual_imbalance':[0,.2],'spot_imbalance':[0,-.3],'variation_active':[True,True]});active,side,_=s.active(panel);assert active.tolist()==[False,True] and side.iloc[1]==-1
def test_fast_midrank_is_strict_prior():
 values=pd.Series([1.,2.,2.,np.nan,3.,1.,4.]);expected=pd.Series([np.nan,np.nan,.75,np.nan,1.,0.,1.]);pd.testing.assert_series_equal(s.strict_prior_midrank(values,3,2),expected)
def test_pinned_registration():assert s.PREREG_SHA=='803c7b3e88eefa69f732e49c938940d588aaeac4966c1240f6fcb78576eac0bf'
