import numpy as np,pandas as pd
from training import build_high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay_support as s
def test_midrank_excludes_current():
 x=pd.Series(np.arange(61,dtype=float));assert s.midrank(x,120,60).iloc[60]==1.
def test_score_state_is_symmetric():
 f=pd.DataFrame({'source_valid':[True]*2,'btc_realized_variation':[1.,2.]})
 for a in s.ALTS:f[f'{a}_quarticity']=[2.,2.];f[f'{a}_realized_variance']=[1.,1.];f[f'{a}_displacement']=[1.,-1.]
 u=s.score_states(f,'no_quarticity_tail');assert u.quarticity_score.tolist()==[1.,-1.];assert u.side.tolist()==[1,-1]
def test_clock_constants_and_registration_are_frozen():
 assert s.P['decision_hours_utc']==[2,10,18] and s.P['path_bars']==96 and s.P['variation_bars']==288;assert s.PREREG_SHA=='9a26d30c718e6ad21d92bb4dd5498293b84664ae6a97bbc1d9be11b98571ffdc'
