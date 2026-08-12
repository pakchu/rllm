import pandas as pd

from training import build_high_volatility_oi_conditioned_late_variation_ignition_relay_support as s


def test_strict_prior_rank_excludes_current():
    ranks = s.strict_prior_midrank(pd.Series(range(181), dtype=float), lookback=270, minimum=180)
    assert ranks.iloc[:180].isna().all() and ranks.iloc[180] == 1.0


def test_primary_requires_fresh_joint_ignition_and_transforms_side():
    frame = pd.DataFrame({
        "source_valid": [True] * 4, "completed_return": [.1,.2,-.2,.2],
        "realized_variation": [1.] * 4, "late_variance_share": [.2] * 4,
        "oi_change": [.1,.1,-.1,-.1], "variation_rank": [.5,.7,.8,.7],
        "late_variance_share_rank": [.8,.8,.8,.6],
    })
    active, side, _ = s.conditions(frame)
    assert active.tolist() == [False, True, False, False]
    assert side.tolist() == [1,1,1,-1]


def test_no_fresh_ignition_is_diagnostic_joint_state():
    frame = pd.DataFrame({"source_valid":[True,True],"completed_return":[.1,-.2],"realized_variation":[1.,1.],"late_variance_share":[.3,.4],"oi_change":[.1,-.1],"variation_rank":[.7,.8],"late_variance_share_rank":[.8,.9]})
    assert s.conditions(frame,"no_fresh_ignition")[0].tolist() == [True,True]
