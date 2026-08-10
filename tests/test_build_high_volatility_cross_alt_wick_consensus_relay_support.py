import numpy as np
import pandas as pd
from training import build_high_volatility_cross_alt_wick_consensus_relay_support as s

def test_wick_consensus_statistics():
 rows=[];times=pd.date_range("2024-01-01",periods=480,freq="1min",tz="UTC")
 for symbol in s.SYMBOLS:
  lower=.002 if symbol!="XRPUSDT" else 0.;upper=0. if symbol!="XRPUSDT" else .001
  for t in times:rows.append((t,symbol,100.,100*np.exp(upper),100*np.exp(-lower),100.))
 block=pd.DataFrame(rows,columns=["ts","symbol","open","high","low","close"]).set_index(["ts","symbol"])
 pressure,equal,breadth,var=s.wick_consensus_statistics(block)
 assert pressure>0 and equal>0 and breadth==5 and var==0

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"consensus_pressure":[.1,-.3,.4,.2,-.5,.4],"strength_rank":[.5,.8,.9,.4,.8,.9],"equal_weight_pressure":[.2,-.2,.3,.1,.4,.2],"consensus_breadth":[5]*6,"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01",periods=6,freq="8h",tz="UTC")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,False,False,True,False] and side[active].tolist()==[-1,-1]
 active,side,_=s.active(panel(),"equal_weight_signed_wick_sum");assert side[active].tolist()==[-1,1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
