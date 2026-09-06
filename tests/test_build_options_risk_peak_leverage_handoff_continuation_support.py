import pandas as pd
from training import build_options_risk_peak_leverage_handoff_continuation_support as s


def test_strict_prior_rank_excludes_current():
 v=pd.Series(range(673),dtype=float);r=s.strict_prior_midrank(v)
 assert r.iloc[:672].isna().all() and r.iloc[672]==1.0


def test_primary_requires_positive_shock_then_cooling():
 f=pd.DataFrame({"source_valid":[True,True],"prior_dvol_body":[.1,.1],"prior_dvol_body_abs_rank":[.8,.8],"dvol_body":[-.1,.1],"dvol_level_rank":[.7,.7],"premium_displacement_rank":[.7,.7],"oi_change":[.1,.1],"premium_displacement":[.01,.01]})
 active,side,_=s.conditions(f)
 assert active.tolist()==[False,False]  # first lacks a valid predecessor; second is not cooling
 assert side.tolist()==[1,1]
