import numpy as np
import pandas as pd
from training import build_high_volatility_elder_impulse_color_transition_relay_support as s

def test_segmented_ewm_resets_after_gap():
 x=pd.Series([1.,2.,3.,4.,5.]);v=pd.Series([True,True,False,True,True]);o=s.segmented_ewm(x,v,.5,2);assert np.isclose(o.iloc[1],1.5) and np.isnan(o.iloc[2]) and np.isnan(o.iloc[3]) and np.isclose(o.iloc[4],4.5)
def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1
def panel():return pd.DataFrame({"source_valid":[True]*8,"color":[0,1,1,0,-1,-1,1,0],"histogram_only_color":[1,1,-1,-1,1,1,-1,-1],"variation_rank":[.8,.8,.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[1] and z.iloc[1]==1 and a.iloc[4] and z.iloc[4]==-1 and not a.iloc[6]
 assert s.active(panel(),"no_variation_gate")[0].iloc[6]
 h,side,_=s.active(panel(),"histogram_slope_only");assert h.iloc[2] and side.iloc[2]==-1 and h.iloc[4] and side.iloc[4]==1
 stale,side,_=s.active(panel(),"one_bar_stale_transition");assert stale.iloc[2] and side.iloc[2]==1
 flipped,side,_=s.active(panel(),"direction_flip");assert flipped.iloc[1] and side.iloc[1]==-1
