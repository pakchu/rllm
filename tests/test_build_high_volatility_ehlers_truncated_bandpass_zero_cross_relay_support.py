import math
import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_truncated_bandpass_zero_cross_relay_support as s

def manual_truncated(values, i, period=20., bandwidth=.1, length=10):
 l1=math.cos(2*math.pi/period);g1=math.cos(bandwidth*2*math.pi/period);s1=1/g1-math.sqrt(1/(g1*g1)-1);state=np.zeros(length+3)
 for count in range(length,0,-1):state[count]=.5*(1-s1)*(values[i-count+1]-values[i-count-1])+l1*(1+s1)*state[count+1]-s1*state[count+2]
 return state[1]

def test_finite_recursion_standard_filter_and_reset():
 prices=pd.Series(100+np.sin(np.arange(80)/3)+np.arange(80)*.02);valid=pd.Series([True]*40+[False]+[True]*39);x=s.bandpasses(prices,valid)
 assert x.truncated_bandpass.first_valid_index()==11 and math.isclose(x.truncated_bandpass.iloc[11],manual_truncated(prices.to_numpy(),11))
 assert x.standard_bandpass.first_valid_index()==2 and x.iloc[40][["truncated_bandpass","standard_bandpass"]].isna().all()
 assert x.run_length.iloc[41:44].tolist()==[1,2,3]
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"standard_cross_side":[0,-1,0,1,0,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})

def test_controls():
 a,z,_=s.active(panel());assert a.iloc[2] and z.iloc[2]==1 and not a.iloc[4]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 raw,side,_=s.active(panel(),"standard_bandpass_zero_cross");assert raw.iloc[1] and side.iloc[1]==-1 and raw.iloc[3] and side.iloc[3]==1
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
 flip,side,_=s.active(panel(),"direction_flip");assert flip.iloc[2] and side.iloc[2]==-1
