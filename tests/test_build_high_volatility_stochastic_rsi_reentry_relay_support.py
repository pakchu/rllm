import numpy as np
import pandas as pd
from training import build_high_volatility_stochastic_rsi_reentry_relay_support as s

def test_stochastic_rsi_is_causal_and_bounded():
 old=dict(s.P);s.P.update(rsi_periods=2,stoch_periods=2)
 try:
  close=pd.Series([10.,11.,10.,12.,11.,13.,12.]);x=s.stochastic_rsi(close,pd.Series([True]*len(close)));assert np.isnan(x.rsi.iloc[1]) and x.rsi.iloc[2]==50 and x.stoch_rsi.iloc[3]==100 and x.stoch_rsi.iloc[4]==0;assert x.stoch_rsi.dropna().between(0,100).all()
 finally:s.P.clear();s.P.update(old)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"stoch_rsi":[10.,15.,30.,50.,90.,75.,50.],"variation_rank":[.8,.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[5]
 outward,side,_=s.active(panel(),"outward_crossing");assert outward.iloc[4] and side.iloc[4]==1
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[3] and side.iloc[3]==1
