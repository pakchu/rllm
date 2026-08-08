import numpy as np
import pandas as pd

from training import build_high_volatility_eth_leadership_relay_support as support


def frame():
    return pd.DataFrame(
        {
            "signal_valid": [True] * 5,
            "btc_late_return": [0.01, -0.01, 0.01, 0.01, -0.01],
            "eth_late_return": [0.02, -0.02, -0.02, 0.02, -0.012],
            "eth_to_btc_absolute_return_ratio": [2.0, 2.0, 2.0, 2.0, 1.2],
            "absolute_eth_return_rank": [0.9, 0.85, 0.9, 0.9, 0.9],
            "variation_rank": [0.9, 0.8, 0.9, 0.4, 0.9],
        }
    )


def test_primary_follows_same_direction_eth_leadership():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False, False]
    assert side[active].tolist() == [1, -1]


def test_controls_remain_diagnostic_only():
    candidate = frame()
    assert support.conditions(candidate, "no_volatility_gate")[0].tolist() == [True, True, False, True, False]
    candidate.loc[1, "absolute_eth_return_rank"] = 0.5
    assert support.conditions(candidate, "no_eth_tail_gate")[0].tolist() == [True, True, False, False, False]
    assert support.conditions(frame(), "no_leadership_ratio_gate")[0].tolist() == [True, True, False, False, True]
    active, side = support.conditions(frame(), "direction_fade")
    assert side[active].tolist() == [-1, 1]


def test_rank_excludes_current_and_freezes_preregistered_history():
    values = pd.Series(np.arange(181, dtype=float))
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[179] != ranks.iloc[179]
    assert ranks.iloc[180] == 1.0
    assert support.strict_prior_midrank.__defaults__ == (270, 180)
