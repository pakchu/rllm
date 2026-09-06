import numpy as np,pandas as pd
from training import build_high_volatility_cross_alt_path_ordered_excursion_dominance_relay_support as s
def test_midrank_excludes_current():
 x=pd.Series(np.arange(61,dtype=float));assert s.midrank(x,120,60).iloc[60]==1.
def test_score_state_is_symmetric():
 f=pd.DataFrame({'source_valid':[True]*2,'btc_realized_variation':[1.,2.]})
 for a in s.ALTS:f[f'{a}_asymmetry']=[.5,-.5]
 u=s.score_states(f,'no_asymmetry_tail');assert u.asymmetry_score.tolist()==[1.,-1.];assert u.side.tolist()==[1,-1]
def test_clock_constants_and_registration_are_frozen():
 assert s.P['decision_hours_utc']==[5,13,21] and s.P['path_bars']==96 and s.P['variation_bars']==288;assert s.PREREG_SHA=='c8359f6523d93e5ccb957223501dd909ad16fd0c42aabc915532b5cf20d3ba64'
