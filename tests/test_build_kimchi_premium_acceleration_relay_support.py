import numpy as np,pandas as pd
from training import build_kimchi_premium_acceleration_relay_support as support

def frame():return pd.DataFrame({"signal_valid":[True]*4,"kimchi_premium":[.01,-.02,.03,-.04],"premium_change":[.01,-.02,.03,-.04],"premium_change_z":[1.2,-1.3,.5,-1.5],"premium_level_z":[1.1,-1.2,.4,-1.4],"btc_realized_variation_rank":[.7,.8,.9,.4]})
def test_primary_direction_and_gates():
 active,side=support.conditions(frame(),"primary");assert active.tolist()==[True,True,False,False];assert side[active].tolist()==[1.,-1.]
def test_controls_are_diagnostic_only():
 f=frame();assert support.conditions(f,"no_volatility_gate")[0].tolist()==[True,True,False,True];assert support.conditions(f,"no_premium_tail")[0].tolist()==[True,True,True,False];active,side=support.conditions(f,"direction_flip");assert side[active].tolist()==[-1.,1.]
def test_causal_statistics_exclude_current():
 x=pd.Series(np.arange(61,dtype=float));z=support.causal_z(x,90,60);assert np.isclose(z.iloc[60],(60-x.iloc[:60].mean())/x.iloc[:60].std(ddof=1));assert support.strict_prior_midrank(x,90,60).iloc[60]==1.
