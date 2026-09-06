import pandas as pd

from training import build_high_volatility_oi_sponsored_late_variation_takeover_relay_support as s


def test_strict_prior_rank_excludes_current():
    ranks=s.strict_prior_midrank(pd.Series(range(181),dtype=float),lookback=270,minimum=180)
    assert ranks.iloc[:180].isna().all() and ranks.iloc[180]==1.0


def test_primary_requires_fresh_takeover_with_oi_expansion():
    frame=pd.DataFrame({"source_valid":[True]*4,"early_return":[.1,.1,.1,.1],"late_return":[.1,-.2,-.2,-.2],"realized_variation":[1.]*4,"late_variance_share":[.3]*4,"oi_change":[.1,.1,.1,-.1],"variation_rank":[.5,.7,.8,.7],"late_variance_share_rank":[.8,.8,.8,.8]})
    active,side,_=s.conditions(frame)
    assert active.tolist()==[False,True,False,False]
    assert side.tolist()==[1,-1,-1,-1]


def test_no_takeover_control_does_not_remove_oi_expansion():
    frame=pd.DataFrame({"source_valid":[True,True],"early_return":[.1,.1],"late_return":[.1,.1],"realized_variation":[1.,1.],"late_variance_share":[.3,.3],"oi_change":[.1,.1],"variation_rank":[.5,.7],"late_variance_share_rank":[.8,.8]})
    assert s.conditions(frame,"no_directional_takeover")[0].tolist()==[False,True]
