import numpy as np
import pandas as pd
from training import build_asian_carry_unwind_concordance_relay_support as support

def frame():
 return pd.DataFrame({"signal_valid":[True]*5,"concordant":[True,True,False,True,True],"aud_pressure":[1.,-1.,1.,1.,-1.],"jpy_pressure":[.5,-.5,-.5,.5,-.5],"risk_pressure":[.75,-.75,.25,.75,-.75],"btc_realized_variation_rank":[.7,.8,.9,.4,.9]})
def test_primary_trades_against_carry_risk_pressure():
 a,s=support.conditions(frame(),"primary");assert a.tolist()==[True,True,False,False,True];assert s[a].tolist()==[-1,1,1]
def test_controls_are_diagnostic_only():
 assert support.CONTROLS==("no_volatility_gate","aud_leg_only","jpy_leg_only","one_session_stale_concordance","direction_flip","forced_long");assert support.conditions(frame(),"no_volatility_gate")[0].tolist()==[True,True,False,True,True];a,s=support.conditions(frame(),"direction_flip");assert s[a].tolist()==[1,-1,-1];a,s=support.conditions(frame(),"forced_long");assert s[a].tolist()==[1,1,1]
def test_causal_rank_excludes_current():
 v=pd.Series(np.arange(61,dtype=float));r=support.strict_prior_midrank(v);assert r.iloc[60]==1.
def test_pair_orientation_and_binding_are_frozen():
 assert support.DOLLAR_MULTIPLIER=={"USDAUD":1.,"USDJPY":1.};assert support.sha(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA;source=support.Path(support.__file__).read_text();assert '"postentry_return_pnl_execution_price_opened":False' in source;assert '"gross9_rows_opened":False' in source
