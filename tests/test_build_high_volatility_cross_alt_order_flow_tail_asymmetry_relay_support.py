import numpy as np
import pandas as pd
from training import build_high_volatility_cross_alt_order_flow_tail_asymmetry_relay_support as s

def test_order_flow_tail_asymmetry_statistics():
 rows=[];times=pd.date_range("2024-01-01",periods=60,freq="1min",tz="UTC")
 flows={"ADAUSDT":-.01,"BNBUSDT":-.005,"DOGEUSDT":0.,"ETHUSDT":.005,"SOLUSDT":.01,"XRPUSDT":.08}
 for symbol in s.SYMBOLS:
  for t in times:rows.append((t,symbol,100.,50.*(flows[symbol]+1.)))
 block=pd.DataFrame(rows,columns=["ts","symbol","quote_asset_volume","taker_buy_quote"]).set_index(["ts","symbol"])
 asymmetry,mean_asymmetry,mass,intensity=s.order_flow_tail_asymmetry_statistics(block)
 assert asymmetry>0 and mean_asymmetry>0 and mass>0 and intensity>0

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"flow_tail_asymmetry":[.1,-.3,.4,.2,-.5,.4],"flow_asymmetry_rank":[.5,.95,.99,.4,.95,.99],"mean_centered_flow_asymmetry":[.2,-.2,.3,.1,.4,.2],"absolute_tail_mass":[.01]*6,"flow_intensity_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01",periods=6,freq="1h",tz="UTC")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,False,False,True,False] and side[active].tolist()==[-1,-1]
 active,side,_=s.active(panel(),"mean_centered_cubic_mass");assert side[active].tolist()==[-1,1]
 x=panel();x.loc[1,"flow_intensity_rank"]=.4;assert s.active(x,"no_flow_intensity_gate")[0].iloc[1]
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(673,dtype=float)));assert np.isnan(ranks.iloc[671]) and ranks.iloc[672]==1.
