import numpy as np,pandas as pd
from training import build_high_volatility_cross_alt_three_bar_flip_cascade_relay_support as s
def test_midrank_excludes_current():
 x=pd.Series(np.arange(16,dtype=float));assert s.midrank(x,20,15).iloc[15]==1.
def test_score_state_is_symmetric():
 f=pd.DataFrame({'source_valid':[True]*2,'btc_realized_variation':[1.,2.]})
 for a in s.ALTS:f[f'{a}_flip_side']=[1,-1]
 u=s.score_states(f,'primary');assert u.positive_flips.tolist()==[6,0];assert u.negative_flips.tolist()==[0,6];assert u.side.tolist()==[1,-1]
def test_clock_constants_and_registration_are_frozen():
 assert s.P['decision_hours_utc']==[0,4,8,12,16,20] and s.P['minimum_flip_alts']==4 and s.P['hold_hours']==6;assert s.PREREG_SHA=='8da080cbd9f607564d1e213c9cdf67b728ef46d786ce32280034c16f815c15c4'
