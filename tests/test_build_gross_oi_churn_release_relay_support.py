import numpy as np,pandas as pd
from training import build_gross_oi_churn_release_relay_support as support
def test_rank_strict():
 x=pd.Series(range(481),dtype=float);r=support.prior_midrank(x);assert np.isnan(r.iloc[479]) and r.iloc[480]==1.
def test_primary_and_flip():
 f=pd.DataFrame({"source_valid":[1,1,1],"late_escape":[.1,.1,.1],"gross_rank":[.5,.7,.7],"net_rank":[.5,.7,.7],"cancellation_rank":[.7,.9,.9],"late_escape_abs_rank":[.7,.7,.7]});a,s=support.active_and_side(f);assert a.tolist()==[False,True,False] and s.tolist()==[1,1,1];_,z=support.active_and_side(f,"direction_flip");assert z.tolist()==[-1,-1,-1]
def test_outcomes_closed():
 s=open(support.__file__).read();assert '"execution_prices_opened":False' in s and '"gross9_rows_opened":False' in s and '"rv20_opened":False' in s
