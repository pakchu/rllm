import numpy as np,pandas as pd
from training import build_high_volatility_cross_alt_realized_skew_spillover_consensus_relay_support as s
def test_midrank_excludes_current():
 x=pd.Series(np.arange(61,dtype=float));assert s.midrank(x,120,60).iloc[60]==1.
def test_score_state_is_symmetric():
 f=pd.DataFrame({'source_valid':[True]*2,'btc_realized_variation':[1.,2.]})
 for a in s.ALTS:f[f'{a}_realized_skew']=[.5,-.5];f[f'{a}_third_moment']=[1.,-1.]
 u=s.score_states(f,'no_skew_tail');assert u.skew_score.tolist()==[1.,-1.];assert u.side.tolist()==[1,-1]
def test_clock_constants_and_registration_are_frozen():
 assert s.P['decision_hours_utc']==[2,14] and s.P['skew_bars']==12 and s.P['variation_bars']==288;assert s.PREREG_SHA=='d2b32b2b020b27cee4e8106a7c3b790ddeb54cc0d099e42e8e6d286b81fead7d'
