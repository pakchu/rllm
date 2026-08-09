import numpy as np,pandas as pd
from training import build_cross_asset_traded_price_jensen_isolation_support as support
def test_rank_strict():
 x=pd.Series(range(1441),dtype=float);r=support.prior_midrank(x);assert np.isnan(r.iloc[1439]) and r.iloc[1440]==1.
def test_primary_and_flip():
 f=pd.DataFrame({"source_valid":[1,1,1],"btc_return":[.1,.1,.1],"btc_jensen_rank":[.8,.95,.95],"isolation_rank":[.8,.95,.95],"close_vwap_displacement":[.1,.1,.1],"close_vwap_abs_rank":[.8,.95,.95]});a,s=support.active_and_side(f);assert a.tolist()==[False,True,False] and s.tolist()==[-1,-1,-1];_,z=support.active_and_side(f,"direction_flip");assert z.tolist()==[1,1,1]
def test_outcomes_closed():
 s=open(support.__file__).read();assert '"execution_prices_opened":False' in s and '"gross9_rows_opened":False' in s and '"rv20_opened":False' in s
