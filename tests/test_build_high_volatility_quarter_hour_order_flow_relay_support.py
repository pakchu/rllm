import numpy as np
import pandas as pd
from training import build_high_volatility_quarter_hour_order_flow_relay_support as s
def test_opening_imbalance():
 assert s.opening_imbalance(100.,60.)==.2 and s.opening_imbalance(100.,40.)==-.2 and np.isnan(s.opening_imbalance(0.,0.))
def panel():
 eligible=[False,True,True,False,True,False]
 return pd.DataFrame({"source_valid":[True]*6,"opening_imbalance":[1.,1.,-1.,-1.,1.,-1.],"imbalance_strength_rank":[.5,.8,.8,.5,.8,.5],"variation_rank":[.8]*6,"eligible":eligible,"onset":[False,True,False,False,True,False],"quarter_anchor":pd.date_range("2024-01-01",periods=6,freq="15min",tz="UTC"),"decision_time":pd.date_range("2024-01-01 00:05",periods=6,freq="15min",tz="UTC"),"feature_available_time":pd.date_range("2024-01-01 00:05",periods=6,freq="15min",tz="UTC")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,False,False,True,False] and side[active].tolist()==[1,1]
 no_onset,_,_=s.active(panel(),"no_onset_requirement");assert no_onset.tolist()==[False,True,True,False,True,False]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(1921,dtype=float)));assert np.isnan(ranks.iloc[1919]) and ranks.iloc[1920]==1.
