import numpy as np,pandas as pd
from training import build_high_volatility_cross_alt_channel_position_migration_relay_support as s
def test_midrank_excludes_current():
 x=pd.Series(np.arange(61,dtype=float));assert s.midrank(x,120,60).iloc[60]==1.
def test_score_state_is_symmetric():
 f=pd.DataFrame({'source_valid':[True]*2,'btc_realized_variation':[1.,2.]})
 for a in s.ALTS:f[f'{a}_migration']=[.2,-.2];f[f'{a}_fixed_prior_channel_migration']=[.1,-.1]
 u=s.score_states(f,'no_migration_tail');assert u.migration_score.tolist()==[.2,-.2];assert u.side.tolist()==[1,-1]
def test_clock_constants_and_registration_are_frozen():
 assert s.P['decision_hours_utc']==[2,14] and s.P['channel_bars']==96 and s.P['migration_lag_bars']==6;assert s.PREREG_SHA=='8f2a1379e27605e0e2a496f55051edc1571dc567f5db5038f067d9a265be8a0d'
