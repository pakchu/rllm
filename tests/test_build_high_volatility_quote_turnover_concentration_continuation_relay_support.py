import numpy as np
import pandas as pd
from training import build_high_volatility_quote_turnover_concentration_continuation_relay_support as support

def test_turnover_statistics_uses_all_96_quote_shares():
 block=pd.DataFrame({"open":np.full(480,100.),"close":np.tile(np.repeat([101.,99.],5),48),"quote_asset_volume":np.concatenate([np.full(5,100.),np.full(475,1.)])})
 hhi,var,weighted,completed=support.turnover_statistics(block)
 assert hhi>1/96
 assert var>0
 assert weighted>0
 assert completed<0

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"concentration_rank":[.5,.85,.9,.4,.9,.9],"variation_rank":[.8,.8,.8,.8,.8,.8],"quote_weighted_return":[.01,.02,.03,-.01,-.02,.02],"completed_return":[.01,.01,.02,-.01,-.01,.01],"feature_available_time":pd.date_range("2024-01-01",periods=6,freq="8h",tz="UTC")})

def test_primary_is_false_to_true_and_uses_weighted_direction():
 active,side,_=support.active(panel(),"primary")
 assert active.tolist()==[False,True,False,False,True,False]
 assert side[active].tolist()==[1,-1]

def test_controls_remain_diagnostic():
 x=panel();x.loc[1,"variation_rank"]=.4
 assert support.active(x,"no_variation_gate")[0].iloc[1]
 active,side,_=support.active(panel(),"direction_flip")
 assert side[active].tolist()==[-1,1]

def test_prior_rank_excludes_current():
 values=pd.Series(np.arange(181,dtype=float));r=support.prior_rank(values)
 assert np.isnan(r.iloc[179]);assert r.iloc[180]==1.
