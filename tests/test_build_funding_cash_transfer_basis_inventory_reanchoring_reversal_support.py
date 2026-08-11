import pandas as pd
from training import build_funding_cash_transfer_basis_inventory_reanchoring_reversal_support as s


def test_strict_prior_rank_excludes_current():
    values=pd.Series(range(181),dtype=float);r=s.strict_prior_midrank(values)
    assert r.iloc[:180].isna().all() and r.iloc[180]==1.0


def test_primary_requires_funding_premium_agreement_and_oi_growth():
    frame=pd.DataFrame({"source_valid":[True,True],"dvol_level_rank":[.7,.7],"premium_mean_rank":[.7,.7],"oi_change":[.1,.1],"funding_rate":[.01,.01],"premium_mean":[.02,-.02]})
    active,side,_=s.conditions(frame)
    assert active.tolist()==[True,False]
    assert side.tolist()==[-1,-1]
