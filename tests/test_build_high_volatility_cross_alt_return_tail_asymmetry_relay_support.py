import numpy as np
import pandas as pd
from training import build_high_volatility_cross_alt_return_tail_asymmetry_relay_support as s

def test_return_tail_asymmetry_statistics():
 rows=[];times=pd.date_range("2024-01-01",periods=480,freq="1min",tz="UTC")
 returns={"BTCUSDT":0.,"ADAUSDT":-.01,"BNBUSDT":-.005,"DOGEUSDT":0.,"ETHUSDT":.005,"SOLUSDT":.01,"XRPUSDT":.08}
 for symbol in s.SYMBOLS:
  for i,t in enumerate(times):
   close=100*np.exp(returns[symbol]) if i==len(times)-1 else 100.
   rows.append((t,symbol,100.,max(100.,close),min(100.,close),close))
 block=pd.DataFrame(rows,columns=["ts","symbol","open","high","low","close"]).set_index(["ts","symbol"])
 asymmetry,mean_asymmetry,mass,var=s.return_tail_asymmetry_statistics(block)
 assert asymmetry>0 and mean_asymmetry>0 and mass>0 and var==0

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"tail_asymmetry":[.1,-.3,.4,.2,-.5,.4],"tail_asymmetry_rank":[.5,.8,.9,.4,.8,.9],"mean_centered_tail_asymmetry":[.2,-.2,.3,.1,.4,.2],"absolute_tail_mass":[.01]*6,"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01",periods=6,freq="8h",tz="UTC")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,False,False,True,False] and side[active].tolist()==[-1,-1]
 active,side,_=s.active(panel(),"mean_centered_cubic_mass");assert side[active].tolist()==[-1,1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
