import numpy as np,pandas as pd
from training import build_daily_late_volume_continuation_relay_support as support

def frame():return pd.DataFrame({"signal_valid":[True]*4,"daily_return":[.04,-.04,.04,.04],"late_return":[.03,-.03,-.03,.03],"late_quote_volume_share":[.4,.5,.5,.4],"variation_rank":[.7,.8,.9,.4]})
def test_primary_direction_and_gates():
 active,side=support.conditions(frame(),"primary");assert active.tolist()==[True,True,False,False];assert side[active].tolist()==[1,-1]
def test_controls_are_diagnostic_only():
 f=frame();assert support.conditions(f,"no_volatility_gate")[0].tolist()==[True,True,False,True];f.loc[1,"late_quote_volume_share"]=.2;assert support.conditions(f,"no_late_volume_gate")[0].tolist()==[True,True,False,False];active,side=support.conditions(frame(),"direction_fade");assert side[active].tolist()==[-1,1]
def test_rank_excludes_current():
 x=pd.Series(np.arange(61,dtype=float));r=support.strict_prior_midrank(x,90,60);assert r.iloc[60]==1.
