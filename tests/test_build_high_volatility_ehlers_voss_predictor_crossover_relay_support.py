import math
import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_voss_predictor_crossover_relay_support as s

def test_voss_recursion_crossover_and_reset():
 prices=pd.Series(100+np.sin(np.arange(100)/4)+np.arange(100)*.01);valid=pd.Series([True]*50+[False]+[True]*49);x=s.voss_predictor(prices,valid);i=9;order=9
 assert x.bandpass.iloc[:5].tolist()==[0.]*5 and x.voss.iloc[:9].tolist()==[0.]*9
 weighted=sum(((count+1)/order)*x.voss.iloc[i-(order-count)] for count in range(order))
 assert math.isclose(x.voss.iloc[i],6*x.bandpass.iloc[i]-weighted) and math.isclose(x.spread.iloc[i],x.voss.iloc[i]-x.bandpass.iloc[i])
 assert x.iloc[50][["bandpass","voss","spread"]].isna().all() and x.run_length.iloc[51:54].tolist()==[1,2,3]
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"voss_zero_side":[0,-1,0,1,0,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})

def test_controls():
 a,z,_=s.active(panel());assert a.iloc[2] and z.iloc[2]==1 and not a.iloc[4]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 raw,side,_=s.active(panel(),"voss_zero_cross");assert raw.iloc[1] and side.iloc[1]==-1 and raw.iloc[3] and side.iloc[3]==1
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
 flip,side,_=s.active(panel(),"direction_flip");assert flip.iloc[2] and side.iloc[2]==-1
