import numpy as np
import pandas as pd
from training import build_high_volatility_parabolic_sar_reversal_relay_support as s

def test_psar_initializes_and_reverses():
 h=pd.Series([10.,11.,12.,11.]);l=pd.Series([8.,9.,10.,7.]);c=pd.Series([9.,10.,11.,8.]);v=pd.Series([True]*4);x=s.parabolic_sar(h,l,c,v,True);assert np.isnan(x.trend.iloc[0]) and x.trend.iloc[1]==1 and x.trend.iloc[3]==-1 and x.reversal.iloc[3] and x.sar.iloc[3]==12

def test_psar_gap_resets():
 h=pd.Series([10.,11.,12.,13.]);l=h-2;c=h-1;v=pd.Series([True,True,False,True]);x=s.parabolic_sar(h,l,c,v);assert np.isnan(x.trend.iloc[3])

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*6,"reversal":[False,True,False,True,True,False],"trend":[1,1,1,-1,1,1],"unclamped_reversal":[False,False,True,False,True,False],"unclamped_trend":[1,1,-1,-1,1,1],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[1] and z.iloc[1]==1 and a.iloc[3] and z.iloc[3]==-1 and not a.iloc[4]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 unclamped,side,_=s.active(panel(),"no_two_bar_clamp");assert unclamped.iloc[2] and side.iloc[2]==-1
 stale,side,_=s.active(panel(),"one_bar_stale_reversal");assert stale.iloc[2] and side.iloc[2]==1
