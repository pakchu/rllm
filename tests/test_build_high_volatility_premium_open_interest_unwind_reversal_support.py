import pandas as pd
from training import build_high_volatility_premium_open_interest_unwind_reversal_support as s


def test_strict_prior_rank_excludes_current():
    values = pd.Series(range(181), dtype=float)
    ranks = s.strict_prior_midrank(values, lookback=270, minimum=180)
    assert ranks.iloc[:180].isna().all()
    assert ranks.iloc[180] == 1.0


def test_primary_requires_all_three_frozen_conditions():
    frame = pd.DataFrame({"source_valid": [True, True], "dvol_level_rank": [.7, .7], "premium_displacement_rank": [.7, .7], "oi_change": [-.1, .1], "premium_displacement": [.01, -.01]})
    active, side, _ = s.conditions(frame)
    assert active.tolist() == [True, False]
    assert side.tolist() == [-1, 1]
