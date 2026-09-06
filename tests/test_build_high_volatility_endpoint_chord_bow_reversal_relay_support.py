import numpy as np
import pandas as pd
from training import build_high_volatility_endpoint_chord_bow_reversal_relay_support as s

def test_bow_statistics_positive_above_chord():
 i=np.arange(481);points=.001*i+0.05*np.sin(np.pi*i/480);prices=100*np.exp(points);open_=prices[:-1];close=prices[1:];block=pd.DataFrame({"open":open_,"high":np.maximum(open_,close),"low":np.minimum(open_,close),"close":close})
 bow,variation=s.bow_statistics(block);assert bow>0 and variation>0

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"normalized_integrated_bow":[-.1,.3,.4,.2,-.5,-.4],"bow_rank":[.5,.8,.9,.4,.8,.9],"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01T05:00:00Z",periods=6,freq="8h")})

def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,False,False,True,False] and side[active].tolist()==[-1,1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()

def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
