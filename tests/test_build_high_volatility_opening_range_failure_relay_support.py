import numpy as np
import pandas as pd
from training import build_high_volatility_opening_range_failure_relay_support as support
def frame():return pd.DataFrame({"source_valid":[True]*5,"exclusive_break":[True,True,True,False,True],"inside_close":[True,True,True,True,False],"opposite_final_return":[True,True,False,True,True],"midpoint_confirmation":[True,False,True,True,True],"primary_state":[True,False,False,False,False],"break_side":[1,-1,1,-1,1],"variation_rank":[.7,.8,.9,.9,.9]})
def test_rank_excludes_current():
 r=support.strict_prior_midrank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(r.iloc[179]);assert r.iloc[180]==1.
def test_primary_and_controls():
 f=frame();a,s=support.conditions(f);assert a.tolist()==[True,False,False,False,False];assert s.tolist()==[-1,1,-1,1,-1];assert support.conditions(f,"inside_close_only")[0].tolist()==[True,True,True,False,False];assert support.conditions(f,"no_midpoint_confirmation")[0].tolist()==[True,True,False,False,False];assert support.conditions(f,"direction_flip")[1].tolist()==[1,-1,1,-1,1]
def test_binding_and_outcome_seal():
 assert support.sha(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA;src=support.Path(support.__file__).read_text();assert '"postentry_return_pnl_execution_price_opened":False' in src;assert '"gross9_rows_opened":False' in src
