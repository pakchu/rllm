import math
import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_reflex_zero_cross_relay_support as s

def test_reflex_formula_crossing_and_reset():
 prices=pd.Series(100+np.sin(np.arange(100)/4)+np.arange(100)*.01);valid=pd.Series([True]*50+[False]+[True]*49);x=s.reflex(prices,valid);i=20
 expected_slope=(x.filtered.iloc[i-20]-x.filtered.iloc[i])/20
 expected_sum=np.mean([(x.filtered.iloc[i]+count*expected_slope)-x.filtered.iloc[i-count] for count in range(1,21)])
 assert math.isclose(x.slope.iloc[i],expected_slope) and math.isclose(x.residual_sum.iloc[i],expected_sum)
 assert math.isclose(x.mean_square.iloc[i],.04*expected_sum**2) and math.isclose(x.reflex.iloc[i],expected_sum/math.sqrt(.04*expected_sum**2))
 assert x.iloc[50][["filtered","slope","residual_sum","mean_square","reflex"]].isna().all() and x.run_length.iloc[51:54].tolist()==[1,2,3]
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"slope_cross_side":[0,-1,0,1,0,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})

def test_controls():
 a,z,_=s.active(panel());assert a.iloc[2] and z.iloc[2]==1 and not a.iloc[4]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 raw,side,_=s.active(panel(),"filtered_slope_zero_cross");assert raw.iloc[1] and side.iloc[1]==-1 and raw.iloc[3] and side.iloc[3]==1
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
 flip,side,_=s.active(panel(),"direction_flip");assert flip.iloc[2] and side.iloc[2]==-1
