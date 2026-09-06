import pandas as pd

from training import build_high_volatility_multi_horizon_trend_ignition_relay_support as s


def test_strict_prior_rank_excludes_current():
    ranks=s.strict_prior_midrank(pd.Series(range(61),dtype=float),lookback=90,minimum=60)
    assert ranks.iloc[:60].isna().all() and ranks.iloc[60]==1.0


def test_primary_requires_variation_onset_and_trend_consensus():
    frame=pd.DataFrame({"source_valid":[True]*4,"return_5d":[.1,.2,.2,-.2],"return_20d":[.2,.3,-.3,-.3],"daily_realized_variation":[1.]*4,"variation_rank":[.5,.7,.8,.4]})
    active,side,_=s.conditions(frame)
    assert active.tolist()==[False,True,False,False]
    assert side.tolist()==[1,1,1,-1]


def test_no_trend_consensus_keeps_variation_onset():
    frame=pd.DataFrame({"source_valid":[True,True],"return_5d":[.1,.2],"return_20d":[.2,-.3],"daily_realized_variation":[1.,1.],"variation_rank":[.5,.7]})
    assert s.conditions(frame,"no_trend_consensus")[0].tolist()==[False,True]
