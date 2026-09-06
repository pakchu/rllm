import numpy as np
import pandas as pd
from training import build_high_volatility_intraday_momentum_index_reentry_relay_support as s

def test_intraday_momentum_uses_candle_bodies():
 old=s.P["imi_periods"];s.P["imi_periods"]=2
 try:
  o=pd.Series([10.,12.,10.]);c=pd.Series([12.,11.,13.]);v=pd.Series([True]*3);x=s.intraday_momentum(o,c,v);assert x.up_body.tolist()==[2.,0.,3.] and x.down_body.tolist()==[0.,1.,0.] and x.imi.iloc[1]==100*2/3 and x.imi.iloc[2]==75
 finally:s.P["imi_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"imi":[20.,25.,40.,50.,80.,65.,50.],"variation_rank":[.8,.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[5]
 outward,side,_=s.active(panel(),"outward_crossing");assert outward.iloc[4] and side.iloc[4]==1
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[3] and side.iloc[3]==1
