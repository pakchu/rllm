import math
import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_quotient_early_onset_relay_support as s

def test_quotient_formula_onset_and_reset():
 prices=pd.Series(100+np.sin(np.arange(240)/8)+np.arange(240)*.01);valid=pd.Series([True]*120+[False]+[True]*119);x=s.quotient_transform(prices,valid);i=20;k=.85
 assert x.high_pass.iloc[:2].tolist()==[0.,0.] and x.filtered.iloc[:2].tolist()==[0.,0.]
 assert math.isclose(x.long_quotient.iloc[i],(x.normalized_roofing.iloc[i]+k)/(k*x.normalized_roofing.iloc[i]+1))
 assert math.isclose(x.short_quotient.iloc[i],(x.normalized_roofing.iloc[i]-k)/(1-k*x.normalized_roofing.iloc[i]))
 assert x.iloc[120][["high_pass","filtered","peak","normalized_roofing","long_quotient","short_quotient"]].isna().all() and x.run_length.iloc[121:124].tolist()==[1,2,3]
 long=pd.Series([-1.,1.,1.,1.]);short=pd.Series([1.,1.,1.,-1.]);assert s.onset_side(long,short).tolist()==[0,1,0,-1]

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"roofing_cross_side":[0,-1,0,1,0,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})

def test_controls():
 a,z,_=s.active(panel());assert a.iloc[2] and z.iloc[2]==1 and not a.iloc[4]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 raw,side,_=s.active(panel(),"normalized_roofing_zero_cross");assert raw.iloc[1] and side.iloc[1]==-1 and raw.iloc[3] and side.iloc[3]==1
 stale,side,_=s.active(panel(),"one_bar_stale_onset");assert stale.iloc[3] and side.iloc[3]==1
 flip,side,_=s.active(panel(),"direction_flip");assert flip.iloc[2] and side.iloc[2]==-1
