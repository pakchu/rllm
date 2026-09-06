import numpy as np
import pandas as pd

from training import build_high_volatility_liquidity_impact_relay_support as support


def test_rank_excludes_current_value():
    values = pd.Series(np.arange(1441, dtype=float))
    ranks = support.strict_prior_midrank(values)
    assert np.isnan(ranks.iloc[1439])
    assert ranks.iloc[1440] == 1.0


def test_primary_is_crossing_continuation():
    frame = pd.DataFrame(
        {
            "signal_valid": [True] * 4,
            "impact_rank": [0.7, 0.85, 0.9, 0.81],
            "raw_return_rank": [0.7, 0.85, 0.9, 0.81],
            "variation_rank": [0.8] * 4,
            "hour_return": [-0.01, 0.02, -0.03, -0.01],
        }
    )
    active, side = support.conditions(frame)
    assert active.tolist() == [False, True, False, False]
    assert side[active].tolist() == [1]


def test_fixed_controls_do_not_change_primary_contract():
    frame = pd.DataFrame(
        {
            "signal_valid": [True, True, True],
            "impact_rank": [0.7, 0.85, 0.7],
            "raw_return_rank": [0.7, 0.75, 0.85],
            "variation_rank": [0.5, 0.5, 0.8],
            "hour_return": [-0.01, 0.02, -0.03],
        }
    )
    assert support.conditions(frame, "no_volatility_gate")[0].tolist() == [False, True, False]
    assert support.conditions(frame, "raw_absolute_return_rank")[0].tolist() == [False, False, True]
    active, side = support.conditions(frame.assign(variation_rank=0.9), "direction_flip")
    assert side[active].tolist() == [-1]
