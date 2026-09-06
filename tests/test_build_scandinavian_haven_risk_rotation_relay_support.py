import numpy as np
import pandas as pd
from training import build_scandinavian_haven_risk_rotation_relay_support as support

def frame():
 return pd.DataFrame({"signal_valid":[True]*5,"risk_pressure":[1.,-1.,.5,1.,-.2],"risk_pressure_rank":[.8,.9,.69,.8,.8],"unoriented_pressure":[1.,-1.,-.5,1.,.2],"unoriented_pressure_rank":[.9,.6,.8,.9,.8],"btc_realized_variation_rank":[.7,.8,.9,.4,.9]})

def test_primary_trades_opposite_extreme_risk_pressure():
 active,side=support.conditions(frame(),"primary");assert active.tolist()==[True,True,False,False,True];assert side[active].tolist()==[-1,1,1]

def test_controls_are_diagnostic_transformations():
 f=frame();assert support.CONTROLS==("no_volatility_gate","no_risk_pressure_tail","no_haven_orientation","one_session_stale_risk_pressure","direction_flip");assert support.conditions(f,"no_volatility_gate")[0].tolist()==[True,True,False,True,True];assert support.conditions(f,"no_risk_pressure_tail")[0].tolist()==[True,True,True,False,True];active,side=support.conditions(f,"no_haven_orientation");assert active.tolist()==[True,False,True,False,True];assert side[active].tolist()==[-1,1,-1];active,side=support.conditions(f,"direction_flip");assert side[active].tolist()==[1,-1,-1]

def test_causal_statistics_exclude_current_value():
 values=pd.Series(np.arange(61,dtype=float));z=support.causal_z(values,lookback=90,minimum=60);expected=(60-values.iloc[:60].mean())/values.iloc[:60].std(ddof=1);assert np.isclose(z.iloc[60],expected);rank=support.strict_prior_midrank(values,lookback=90,minimum=60);assert rank.iloc[60]==1.

def test_risk_orientation_is_frozen():
 assert support.RISK_ORIENTATION=={"USDSEK":1.,"USDCHF":-1.,"USDJPY":-1.}

def test_builder_binds_preregistration_and_seals_outcomes():
 assert support.sha(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA;s=support.Path(support.__file__).read_text();assert '"postentry_return_pnl_execution_price_opened":False' in s;assert '"gross9_rows_opened":False' in s
