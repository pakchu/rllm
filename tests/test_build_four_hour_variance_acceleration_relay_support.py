import numpy as np
import pandas as pd

from training import build_four_hour_variance_acceleration_relay_support as support


def frame():
    return pd.DataFrame(
        {
            "signal_valid": [True] * 4,
            "first_return": [0.01, -0.01, 0.01, 0.01],
            "second_return": [0.02, -0.02, -0.02, 0.02],
            "second_to_first_variation": [2.0, 1.5, 2.0, 2.0],
            "variation_rank": [0.9, 0.8, 0.9, 0.4],
        }
    )


def test_primary_follows_confirmed_acceleration_direction():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False]
    assert side[active].tolist() == [1, -1]


def test_controls_remain_diagnostic_only():
    candidate = frame()
    assert support.conditions(candidate, "no_volatility_gate")[0].tolist() == [True, True, False, True]
    candidate.loc[1, "second_to_first_variation"] = 1.0
    assert support.conditions(candidate, "no_acceleration_gate")[0].tolist() == [True, True, False, False]
    active, side = support.conditions(frame(), "direction_fade")
    assert side[active].tolist() == [-1, 1]


def test_rank_excludes_current_observation():
    values = pd.Series(np.arange(121, dtype=float))
    ranks = support.strict_prior_midrank(values, 180, 120)
    assert ranks.iloc[119] != ranks.iloc[119]
    assert ranks.iloc[120] == 1.0
