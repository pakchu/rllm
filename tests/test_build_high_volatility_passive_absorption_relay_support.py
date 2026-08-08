import numpy as np
import pandas as pd

from training import build_high_volatility_passive_absorption_relay_support as support


def frame():
    return pd.DataFrame(
        {
            "signal_valid": [True] * 5,
            "late_return": [0.015, -0.015, 0.03, 0.015, -0.02],
            "late_taker_imbalance": [-0.2, 0.2, 0.2, -0.2, 0.05],
            "variation_rank": [0.9, 0.8, 0.9, 0.4, 0.9],
        }
    )


def test_primary_follows_price_resilient_against_aggressive_flow():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False, False]
    assert side[active].tolist() == [1, -1]


def test_controls_remain_diagnostic_only():
    candidate = frame()
    assert support.conditions(candidate, "no_volatility_gate")[0].tolist() == [True, True, False, True, False]
    assert support.conditions(candidate, "no_flow_magnitude_gate")[0].tolist() == [True, True, False, False, True]
    assert support.conditions(candidate, "no_contradiction_gate")[0].tolist() == [True, True, True, False, False]
    active, side = support.conditions(frame(), "direction_fade")
    assert side[active].tolist() == [-1, 1]


def test_rank_excludes_current_and_freezes_preregistered_history():
    values = pd.Series(np.arange(181, dtype=float))
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[179] != ranks.iloc[179]
    assert ranks.iloc[180] == 1.0
    assert support.strict_prior_midrank.__defaults__ == (270, 180)
