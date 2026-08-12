import numpy as np
import pandas as pd

from training import build_high_volatility_top_two_causal_expert_concordance_relay_support as s


def test_rank_excludes_current():
    ranks=s.rank(pd.Series(np.arange(181,dtype=float)))
    assert ranks.iloc[:180].isna().all() and ranks.iloc[180]==1.0


def test_expert_sides_use_4h_and_24h():
    assert s.sides(.1,-.2).tolist()==[1,-1,-1,1]


def test_primary_requires_top_two_concordance():
    frame=pd.DataFrame({"ranking_valid":[True,True],"concordance":[False,True],"top_side":[-1,1],"common_side":[0,1],"variation_rank":[.8,.8]})
    active,side=s.conditions(frame,"primary")
    assert active.tolist()==[False,True] and side.tolist()==[0,1]
    top_active,top_side=s.conditions(frame,"no_top_two_concordance")
    assert top_active.all() and top_side.tolist()==[-1,1]
