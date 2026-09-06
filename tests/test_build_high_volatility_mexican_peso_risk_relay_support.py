import numpy as np
import pandas as pd
from training import build_high_volatility_mexican_peso_risk_relay_support as s
def frame():return pd.DataFrame({'signal_valid':[True]*4,'peso_risk_direction':[1,-1,1,-1],'shock_rank':[.8,.9,.7,.8],'btc_realized_variation_rank':[.7,.8,.9,.4]})
def test_primary_follows_canonical_peso_risk():
 active,side=s.conditions(frame(),'primary');assert active.tolist()==[True,True,False,False] and side[active].tolist()==[1,-1]
def test_controls_are_diagnostic():
 assert s.conditions(frame(),'no_shock_tail')[0].tolist()==[True,True,True,False];active,side=s.conditions(frame(),'direction_flip');assert side[active].tolist()==[-1,1]
def test_causal_rank_excludes_current():
 values=pd.Series(np.arange(61,dtype=float));assert s.strict_prior_midrank(values).iloc[60]==1.
def test_pinned_registration():assert s.PREREG_SHA=='c9cc539b886240748f858c8fe60cf44f179e20d834623a8900b06db0c1af63db'
