import numpy as np,pandas as pd
from training import build_high_volatility_cross_alt_variance_shock_propagation_relay_support as s
def test_midrank_excludes_current():
 x=pd.Series(np.arange(61,dtype=float));assert s.midrank(x,120,60).iloc[60]==1.
def test_score_state_is_symmetric():
 f=pd.DataFrame({'source_valid':[True]*2,'btc_realized_variation':[1.,2.]})
 for a in s.ALTS:f[f'{a}_realized_variance']=[1.,1.];f[f'{a}_displacement']=[1.,-1.]
 u=s.score_states(f,'no_shock_tail');assert u.shock_score.tolist()==[1.,-1.];assert u.side.tolist()==[1,-1]
def test_clock_constants_and_registration_are_frozen():
 assert s.P['decision_hours_utc']==[2,14] and s.P['shock_bars']==12 and s.P['variation_bars']==288;assert s.PREREG_SHA=='dbc0eadc608cd8008a5ab029cface8474d868631a3ce1f37c3f3f5ce3856f560'
