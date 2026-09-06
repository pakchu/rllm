import numpy as np
import pandas as pd
from training import build_hkd_peg_pressure_rotation_relay_support as support

def frame():
 return pd.DataFrame({"signal_valid":[True]*5,"peg_pressure_change":[.01,-.02,.01,.02,-.01],"pressure_magnitude_rank":[.8,.9,.69,.8,.8],"btc_realized_variation_rank":[.7,.8,.9,.4,.9]})

def test_primary_trades_opposite_extreme_peg_pressure():
 active,side=support.conditions(frame(),"primary");assert active.tolist()==[True,True,False,False,True];assert side[active].tolist()==[-1,1,1]

def test_controls_are_diagnostic_transformations():
 f=frame();assert support.CONTROLS==("no_volatility_gate","no_pressure_tail","one_session_stale_pressure","direction_flip");assert support.conditions(f,"no_volatility_gate")[0].tolist()==[True,True,False,True,True];assert support.conditions(f,"no_pressure_tail")[0].tolist()==[True,True,True,False,True];active,side=support.conditions(f,"direction_flip");assert side[active].tolist()==[1,-1,-1]

def test_causal_rank_excludes_current_value():
 values=pd.Series(np.arange(61,dtype=float));rank=support.strict_prior_midrank(values,lookback=90,minimum=60);assert rank.iloc[60]==1.0

def test_builder_binds_preregistration_and_seals_outcomes():
 assert support.SYMBOL=="USDHKD";assert support.sha(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA;s=support.Path(support.__file__).read_text();assert '"postentry_return_pnl_execution_price_opened":False' in s;assert '"gross9_rows_opened":False' in s
