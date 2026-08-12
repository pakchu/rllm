import numpy as np
import pandas as pd
from training import build_high_volatility_safehaven_intraday_semivariance_relay_support as s

def frame():
 return pd.DataFrame({'signal_valid':[True]*5,'semivariance_imbalance':[.4,-.3,0.,.2,-.5],'JPY_imbalance':[-.1,.2,.3,-.4,.1],'CHF_imbalance':[.2,-.1,-.2,.3,-.4],'imbalance_rank':[.8,.9,.95,.6,.8],'btc_realized_variation_rank':[.7,.8,.9,.9,.4]})
def test_primary_follows_pooled_safehaven_pressure():
 active,side=s.conditions(frame(),'primary');assert active.tolist()==[True,True,False,False,False];assert side[active].tolist()==[-1.,1.]
def test_controls_are_diagnostic():
 active,_=s.conditions(frame(),'no_imbalance_tail');assert active.tolist()==[True,True,False,True,False]
 active,side=s.conditions(frame(),'JPY_only');assert side[active].tolist()==[1.,-1.,-1.]
 active,side=s.conditions(frame(),'forced_long');assert side[active].tolist()==[1,1]
def test_causal_local_ranks_exclude_current():
 values=pd.Series(np.arange(41,dtype=float));assert s.strict_prior_midrank(values,lookback=60,minimum=40).iloc[40]==1.
 local=pd.Series(np.arange(16,dtype=float));assert s.strict_prior_midrank(local,lookback=20,minimum=15).iloc[15]==1.
def test_registration_is_frozen():assert s.PREREG_SHA=='263584f62d7e55dbd1f414b6dfa8d628d691374a6b2b1176b8ec9b0c303e51d2'
