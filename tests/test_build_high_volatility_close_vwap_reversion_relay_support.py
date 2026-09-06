import numpy as np
import pandas as pd

from training import build_high_volatility_close_vwap_reversion_relay_support as support


def frame():
    return pd.DataFrame(
        {
            "signal_valid": [True] * 4,
            "normalized_displacement": [-.3,.3,.1,-.3],
            "variation_rank": [0.9, 0.8, 0.9, 0.4],
        }
    )


def test_primary_fades_close_vwap_displacement():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False]
    assert side[active].tolist() == [1, -1]


def test_controls_remain_diagnostic_only():
    candidate = frame()
    assert support.conditions(candidate, "no_volatility_gate")[0].tolist() == [True, True, False, True]
    candidate.loc[1, "normalized_displacement"] = .1
    assert support.conditions(candidate, "no_displacement_gate")[0].tolist() == [True, True, True, False]
    active, side = support.conditions(frame(), "direction_follow")
    assert side[active].tolist() == [-1, 1]


def test_rank_excludes_current_observation():
    values = pd.Series(np.arange(121, dtype=float))
    ranks = support.strict_prior_midrank(values, 180, 120)
    assert ranks.iloc[119] != ranks.iloc[119]
    assert ranks.iloc[120] == 1.0
