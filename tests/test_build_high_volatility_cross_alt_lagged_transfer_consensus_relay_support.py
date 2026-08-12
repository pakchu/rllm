import numpy as np,pandas as pd
from training import build_high_volatility_cross_alt_lagged_transfer_consensus_relay_support as s
def test_midrank_excludes_current():
 x=pd.Series(np.arange(61,dtype=float));assert s.midrank(x,120,60).iloc[60]==1.
def test_score_state_is_symmetric_and_causal():
 f=pd.DataFrame({'source_valid':[True]*2,'btc_realized_variation':[1.,2.]})
 for a in s.ALTS:f[f'{a}_beta']=[1.,-1.];f[f'{a}_latest_return']=[1.,1.]
 u=s.score_states(f,'no_beta_tail');assert u.transfer_score.tolist()==[1.,-1.];assert u.side.tolist()==[1,-1]
def test_clock_constants_and_registration_are_frozen():
 assert s.P['decision_hours_utc']==[2,10,18] and s.P['beta_bars']==288;assert s.PREREG_SHA=='0b32ed107a661dbec816ad117e047b6ae75783a6026216c2dfd8b6444bd98943'
