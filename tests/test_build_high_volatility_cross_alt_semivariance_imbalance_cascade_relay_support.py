import numpy as np,pandas as pd
from training import build_high_volatility_cross_alt_semivariance_imbalance_cascade_relay_support as s
def test_midrank_excludes_current():
 x=pd.Series(np.arange(61,dtype=float));assert s.midrank(x,120,60).iloc[60]==1.
def test_score_state_is_symmetric():
 f=pd.DataFrame({'source_valid':[True]*2,'btc_realized_variation':[1.,2.]})
 for a in s.ALTS:f[f'{a}_semivariance_imbalance']=[.5,-.5]
 u=s.score_states(f,'no_semivariance_imbalance_tail');assert u.semivariance_imbalance_score.tolist()==[1.,-1.];assert u.side.tolist()==[1,-1]
def test_clock_constants_and_registration_are_frozen():
 assert s.P['decision_hours_utc']==[1,9,17] and s.P['path_bars']==72 and s.P['variation_bars']==288;assert s.PREREG_SHA=='bfa24ad3de932076ebe261bd54b482d21f6c3384bc69e30e4ce7ab9fb85bea3a'
