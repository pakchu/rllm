import numpy as np,pandas as pd
from training import build_daily_range_close_momentum_relay_support as support

def frame():return pd.DataFrame({"signal_valid":[True]*4,"daily_open":[100.]*4,"daily_close":[110.,90.,105.,95.],"close_location":[.9,.1,.6,.4],"range_rank":[.7,.8,.9,.4]})
def test_primary_direction_and_gates():
 active,side=support.conditions(frame(),"primary");assert active.tolist()==[True,True,False,False];assert side[active].tolist()==[1,-1]
def test_controls_are_diagnostic_only():
 f=frame();assert support.conditions(f,"no_range_gate")[0].tolist()==[True,True,False,False];assert support.conditions(f,"no_close_location_gate")[0].tolist()==[True,True,True,False];active,side=support.conditions(f,"direction_flip");assert side[active].tolist()==[-1,1]
def test_rank_excludes_current():
 x=pd.Series(np.arange(61,dtype=float));r=support.strict_prior_midrank(x,90,60);assert r.iloc[60]==1.
