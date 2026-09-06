import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_instantaneous_trendline_crossover_relay_support as s

def test_instantaneous_trendline_seed_recursion_and_reset():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 old=dict(s.P);s.P.update(alpha=.07,initialization_observations=6)
 try:
  prices=pd.Series(np.arange(10.,20.));x=s.instantaneous_trendline(prices,pd.Series([True]*10));assert np.isnan(x.itrend.iloc[1]) and x.itrend.iloc[2]==11 and np.isfinite(x.itrend.iloc[6]) and np.isfinite(x.trigger.iloc[6])
  gap=s.instantaneous_trendline(prices,pd.Series([True]*5+[False]+[True]*4));assert gap.itrend.iloc[5]!=gap.itrend.iloc[5] and gap.itrend.iloc[7]!=gap.itrend.iloc[7] and np.isfinite(gap.itrend.iloc[8])
 finally:s.P.clear();s.P.update(old)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"price_cross_side":[0,0,1,0,-1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
