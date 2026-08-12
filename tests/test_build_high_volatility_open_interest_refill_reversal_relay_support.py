import pandas as pd

from training import build_high_volatility_open_interest_refill_reversal_relay_support as s


def test_strict_prior_rank_excludes_current():
    values = pd.Series(range(181), dtype=float)
    ranks = s.strict_prior_midrank(values, lookback=270, minimum=180)
    assert ranks.iloc[:180].isna().all()
    assert ranks.iloc[180] == 1.0


def test_primary_requires_frozen_contraction_refill_transition():
    frame = pd.DataFrame(
        {
            "source_valid": [True, True, True, True],
            "dvol_level_rank": [.7] * 4,
            "premium_displacement_rank": [.7] * 4,
            "prior_oi_change": [-.1, -.1, .1, -.1],
            "current_oi_change": [.03, .02, .03, -.03],
            "refill_fraction": [.3, .2, .3, .3],
            "premium_displacement": [.01, -.01, .01, -.01],
        }
    )
    active, side, _ = s.conditions(frame)
    assert active.tolist() == [True, False, False, False]
    assert side.tolist() == [-1, 1, -1, 1]


def test_no_refill_fraction_control_keeps_transition_but_not_threshold():
    frame = pd.DataFrame(
        {
            "source_valid": [True],
            "dvol_level_rank": [.7],
            "premium_displacement_rank": [.7],
            "prior_oi_change": [-.1],
            "current_oi_change": [.02],
            "refill_fraction": [.2],
            "premium_displacement": [-.01],
        }
    )
    assert s.conditions(frame, "no_refill_fraction")[0].tolist() == [True]
