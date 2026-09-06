import numpy as np
import pandas as pd
from training import build_high_volatility_fx_safehaven_cyclical_barbell_relay_support as s

def frame():
 return pd.DataFrame({'signal_valid':[True]*5,'common_risk_direction':[1,-1,0,0,-1],'positive_qualifying_pairs':[3,0,2,3,0],'negative_qualifying_pairs':[0,4,0,1,3],'common_dollar_side':[-1,1,1,0,-1],'agreeing_pairs':[3,4,2,3,3],'median_absolute_pair_z':[.8,1.2,.6,.7,.9],'btc_realized_variation_rank':[.7,.8,.9,.9,.4]})
def test_primary_follows_fx_risk_barbell():
 active,side=s.conditions(frame(),'primary');assert active.tolist()==[True,True,False,False,False];assert side[active].tolist()==[1,-1]
def test_barbell_controls_are_diagnostic():
 active,side=s.conditions(frame(),'two_of_four_barbell');assert active.tolist()==[True,True,True,False,False]
 active,side=s.conditions(frame(),'direction_flip');assert side[active].tolist()==[-1,1]
 active,side=s.conditions(frame(),'forced_long');assert side[active].tolist()==[1,1]
def test_causal_statistics_exclude_current():
 values=pd.Series(np.arange(61,dtype=float));z=s.causal_z(values);expected=(60-values.iloc[:60].mean())/values.iloc[:60].std(ddof=1);assert np.isclose(z.iloc[60],expected);assert s.strict_prior_midrank(values).iloc[60]==1.
def test_canonical_risk_orientation_is_frozen():
 assert s.RISK_MULTIPLIER=={'USDJPY':1.,'USDCHF':1.,'USDMXN':-1.,'USDAUD':-1.}
def test_pinned_registration():assert s.PREREG_SHA=='574cb9aae93689145fb03ae00a73ee47f89e473dae0877191a4a2c79dba866ab'
