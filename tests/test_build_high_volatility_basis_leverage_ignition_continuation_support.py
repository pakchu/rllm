import pandas as pd
from training import build_high_volatility_basis_leverage_ignition_continuation_support as s


def test_strict_prior_rank_excludes_current():
    values=pd.Series(range(673),dtype=float); ranks=s.strict_prior_midrank(values)
    assert ranks.iloc[:672].isna().all(); assert ranks.iloc[672]==1.0


def test_primary_is_false_to_true_ignition_onset():
    frame=pd.DataFrame({"source_valid":[True]*4,"dvol_level_rank":[.7]*4,"dvol_open":[10]*4,"dvol_close":[11,11,9,11],"premium_displacement_rank":[.7]*4,"oi_change":[.1]*4,"premium_displacement":[.01]*4})
    active,side,_=s.conditions(frame)
    assert active.tolist()==[False,False,False,True]
    assert side.tolist()==[1,1,1,1]
