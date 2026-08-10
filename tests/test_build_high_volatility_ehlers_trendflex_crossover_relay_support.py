import math
import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_trendflex_crossover_relay_support as s

def test_trendflex_recursion_crossover_and_reset():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 prices=pd.Series(100+np.sin(np.arange(180)/4)+np.arange(180)*.01)
 valid=pd.Series([True]*90+[False]+[True]*89)
 x=s.trendflex(prices,valid,20)
 assert x["filter"].iloc[:3].tolist()==prices.iloc[:3].tolist()
 i=20;expected=float(np.mean(x["filter"].iloc[i]-x["filter"].iloc[i-20:i]))
 assert math.isclose(x["sum"].iloc[i],expected) and math.isclose(x.ms.iloc[i],.04*expected**2)
 assert x.iloc[90][["filter","sum","ms","trendflex"]].isna().all()
 assert x.run_length.iloc[91:94].tolist()==[1,2,3]

def test_pair_uses_published_lengths_and_prior_rank_excludes_current():
 prices=pd.Series(100+np.sin(np.arange(220)/5)+np.arange(220)*.02)
 x=s.trendflex_pair(prices,pd.Series(True,index=prices.index))
 assert x.fast_trendflex.first_valid_index()==20 and x.slow_trendflex.first_valid_index()==50
 assert x.entry_side.equals(s.cross_side(x.fast_trendflex-x.slow_trendflex))
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)))
 assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"fast_zero_side":[0,-1,0,1,0,0],"slow_zero_side":[0,0,-1,0,1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})

def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 raw,side,_=s.active(panel(),"fast_trendflex_zero_cross");assert raw.iloc[1] and side.iloc[1]==-1 and raw.iloc[3] and side.iloc[3]==1
 slow,side,_=s.active(panel(),"slow_trendflex_zero_cross");assert slow.iloc[2] and side.iloc[2]==-1
 flip,side,_=s.active(panel(),"direction_flip");assert flip.iloc[2] and side.iloc[2]==-1
