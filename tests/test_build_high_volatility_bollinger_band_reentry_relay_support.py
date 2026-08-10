import numpy as np
import pandas as pd
from training import build_high_volatility_bollinger_band_reentry_relay_support as s

def test_segmented_ewm_resets_after_gap():
 x=pd.Series([1.,2.,3.,4.,5.]);v=pd.Series([True,True,False,True,True]);o=s.segmented_ewm(x,v,.5,2);assert np.isclose(o.iloc[1],1.5) and np.isnan(o.iloc[2]) and np.isnan(o.iloc[3]) and np.isclose(o.iloc[4],4.5)
def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1
def panel():return pd.DataFrame({"source_valid":[True]*7,"bar_close":[7.,10.,10.,13.,10.,7.,10.],"upper":[12.]*7,"lower":[8.]*7,"ema_upper":[11.]*7,"ema_lower":[9.]*7,"variation_rank":[.8,.8,.8,.8,.4,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[1] and z.iloc[1]==1 and not a.iloc[4] and a.iloc[6] and z.iloc[6]==1
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 ema,side,_=s.active(panel(),"exponential_center");assert ema.iloc[1] and side.iloc[1]==1 and ema.iloc[6] and side.iloc[6]==1
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[2] and side.iloc[2]==1
 flipped,side,_=s.active(panel(),"direction_flip");assert flipped.iloc[1] and side.iloc[1]==-1
