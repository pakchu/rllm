import numpy as np
import pandas as pd
from training import build_high_volatility_keltner_channel_breakout_relay_support as s
def test_segmented_ewm_resets_after_gap():
 x=pd.Series([1.,2.,3.,4.,5.]);valid=pd.Series([True,True,False,True,True]);out=s.segmented_ewm(x,valid,.5,2);assert np.isclose(out.iloc[1],1.5) and np.isnan(out.iloc[2]) and np.isnan(out.iloc[3]) and np.isclose(out.iloc[4],4.5)
def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1
def panel():return pd.DataFrame({"source_valid":[True]*7,"bar_close":[10.,13.,12.,7.,13.,12.,12.],"upper":[12.]*7,"lower":[8.]*7,"sma_upper":[11.]*7,"sma_lower":[9.]*7,"variation_rank":[.8,.8,.8,.8,.4,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[1] and z.iloc[1]==1 and a.iloc[3] and z.iloc[3]==-1 and not a.iloc[4]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 simple,side,_=s.active(panel(),"simple_moving_center");assert simple.iloc[1] and side.iloc[1]==1
 stale,side,_=s.active(panel(),"one_bar_stale_breakout");assert stale.iloc[2] and side.iloc[2]==1
 flipped,side,_=s.active(panel(),"direction_flip");assert flipped.iloc[1] and side.iloc[1]==-1
