import pandas as pd

from training import build_high_volatility_inventory_conditioned_variation_onset_relay_support as s


def test_strict_prior_rank_excludes_current():
    ranks = s.strict_prior_midrank(pd.Series(range(181), dtype=float), lookback=270, minimum=180)
    assert ranks.iloc[:180].isna().all() and ranks.iloc[180] == 1.0


def test_primary_requires_below_to_high_onset_and_transforms_side():
    frame = pd.DataFrame({
        "source_valid": [True] * 4,
        "completed_return": [.1, .2, -.2, .2],
        "realized_variation": [1.] * 4,
        "oi_change": [.1, .1, -.1, -.1],
        "variation_rank": [.5, .7, .8, .4],
    })
    active, side, _ = s.conditions(frame)
    assert active.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, 1, -1]


def test_ignore_inventory_sign_control_follows_completed_return():
    frame = pd.DataFrame({"source_valid":[True,True],"completed_return":[.1,-.2],"realized_variation":[1.,1.],"oi_change":[.1,-.1],"variation_rank":[.5,.7]})
    active, side, _ = s.conditions(frame, "ignore_inventory_sign_follow_return")
    assert active.tolist() == [False, True] and side.tolist() == [1, -1]
