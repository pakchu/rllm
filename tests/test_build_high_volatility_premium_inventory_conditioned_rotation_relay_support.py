import pandas as pd

from training import build_high_volatility_premium_inventory_conditioned_rotation_relay_support as s


def test_strict_prior_rank_excludes_current():
    ranks = s.strict_prior_midrank(pd.Series(range(181), dtype=float), lookback=270, minimum=180)
    assert ranks.iloc[:180].isna().all() and ranks.iloc[180] == 1.0


def test_primary_uses_oi_sign_as_direction_transform():
    frame = pd.DataFrame({
        "source_valid": [True] * 4, "dvol_level_rank": [.7] * 4, "premium_displacement_rank": [.7] * 4,
        "prior_premium_displacement": [-.1, .1, -.1, -.1],
        "current_premium_displacement": [.2, -.2, -.2, .2],
        "current_oi_change": [.01, -.01, .01, -.01],
    })
    active, side, _ = s.conditions(frame)
    assert active.tolist() == [True, True, False, True]
    assert side.tolist() == [1, 1, -1, -1]


def test_ignore_inventory_sign_is_diagnostic_follow_premium():
    frame = pd.DataFrame({
        "source_valid": [True], "dvol_level_rank": [.7], "premium_displacement_rank": [.7],
        "prior_premium_displacement": [.1], "current_premium_displacement": [-.2], "current_oi_change": [-.01],
    })
    active, side, _ = s.conditions(frame, "ignore_inventory_sign_follow_premium")
    assert active.tolist() == [True] and side.tolist() == [-1]
