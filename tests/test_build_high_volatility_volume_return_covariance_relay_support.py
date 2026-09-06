import numpy as np
import pandas as pd
from training import build_high_volatility_volume_return_covariance_relay_support as support

def frame():
 return pd.DataFrame({"source_valid":[True]*5,"realized_variation":[1.]*5,"volume_return_correlation":[.2,-.3,.1,.4,-.2],"unweighted_return":[2.,-3.,-1.,4.,2.],"variation_rank":[.8,.7,.9,.5,.8],"absolute_correlation_rank":[.8,.9,.6,.9,.8]})

def test_primary_follows_extreme_covariance_in_high_variation():
 a,s=support.conditions(frame(),"primary");assert a.tolist()==[True,True,False,False,True];assert s[a].tolist()==[1,-1,-1]

def test_diagnostic_controls_are_frozen():
 assert support.CONTROLS==("no_volatility_gate","no_correlation_tail","unweighted_return","one_day_stale_features","direction_flip","forced_long");assert support.conditions(frame(),"no_volatility_gate")[0].tolist()==[True,True,False,True,True];assert support.conditions(frame(),"no_correlation_tail")[0].tolist()==[True,True,True,False,True];a,s=support.conditions(frame(),"unweighted_return");assert s[a].tolist()==[1,-1,1];a,s=support.conditions(frame(),"direction_flip");assert s[a].tolist()==[-1,1,1];a,s=support.conditions(frame(),"forced_long");assert s[a].tolist()==[1,1,1]

def test_strict_prior_midrank_excludes_current():
 v=pd.Series(np.arange(61,dtype=float));r=support.strict_prior_midrank(v);assert np.isnan(r.iloc[59]);assert r.iloc[60]==1.

def test_builder_binds_preregistration_and_seals_outcomes():
 assert support.sha256(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA;source=support.Path(support.__file__).read_text();assert '"postentry_return_pnl_execution_price_opened": False' in source;assert '"gross9_rows_opened": False' in source
