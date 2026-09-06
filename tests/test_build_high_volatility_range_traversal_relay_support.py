import numpy as np,pandas as pd
from training import build_high_volatility_range_traversal_relay_support as support

def frame():return pd.DataFrame({"signal_valid":[True]*4,"daily_open":[100.]*4,"daily_close":[110.,90.,105.,95.],"close_location":[.8,.2,.6,.4],"range_rank":[.7,.8,.9,.4],"first_high_time":pd.to_datetime(["2024-01-01T12:00Z","2024-01-01T01:00Z","2024-01-01T12:00Z","2024-01-01T01:00Z"]),"first_low_time":pd.to_datetime(["2024-01-01T01:00Z","2024-01-01T12:00Z","2024-01-01T01:00Z","2024-01-01T12:00Z"])})
def test_primary_direction_and_gates():
 active,side=support.conditions(frame(),"primary");assert active.tolist()==[True,True,False,False];assert side[active].tolist()==[1,-1]
def test_controls_are_diagnostic_only():
 f=frame();assert support.conditions(f,"no_range_gate")[0].tolist()==[True,True,False,False];assert support.conditions(f,"no_close_location")[0].tolist()==[True,True,True,False];active,side=support.conditions(f,"direction_flip");assert side[active].tolist()==[-1,1]
def test_rank_excludes_current():
 x=pd.Series(np.arange(181,dtype=float));r=support.strict_prior_midrank(x);assert r.iloc[180]==1.

def test_preregistration_is_hash_bound():assert support.PREREG_SHA=="c931469ea605b5b6bc036acc556e860adb8130e3fae3455ef22ceae1ec531014"
