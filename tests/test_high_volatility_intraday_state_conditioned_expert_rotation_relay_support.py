import numpy as np,pandas as pd
from training import build_high_volatility_intraday_state_conditioned_expert_rotation_relay_support as s

def test_rank_excludes_current():
 ranks=s.rank(pd.Series(np.arange(181,dtype=float)));assert ranks.iloc[:180].isna().all() and ranks.iloc[180]==1.

def test_intraday_expert_sides():assert s.sides(.1,-.2).tolist()==[1,-1,-1,1]

def test_primary_requires_positive_rotation_and_variation():
 x=pd.DataFrame({'ranking_valid':[True,True],'rotation':[False,True],'positive_score':[True,True],'winner_side':[-1,1],'variation_rank':[.8,.8]})
 active,side=s.conditions(x,'primary');assert active.tolist()==[False,True] and side.tolist()==[-1,1]
