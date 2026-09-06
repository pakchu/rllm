import numpy as np,pandas as pd
from training import build_high_volatility_cross_alt_flow_impact_escalation_relay_support as s
def test_midrank_excludes_current():
 x=pd.Series(np.arange(61,dtype=float));assert s.midrank(x,120,60).iloc[60]==1.
def test_score_state_is_symmetric():
 f=pd.DataFrame({'source_valid':[True]*2,'btc_realized_variation':[1.,2.]})
 for a in s.ALTS:f[f'{a}_escalation']=[2.,2.];f[f'{a}_direction']=[1,-1]
 u=s.score_states(f,'no_escalation_tail');assert u.escalation_score.tolist()==[1.,-1.];assert u.side.tolist()==[1,-1]
def test_clock_constants_and_registration_are_frozen():
 assert s.P['decision_hours_utc']==[3,11,19] and s.P['block_bars']==96 and s.P['half_bars']==48 and s.P['variation_bars']==288;assert s.PREREG_SHA=='458073b97635405ab1d350c5162b9d096ea1f5eb9b84a1cea2af5b8d9133c78f'
