import numpy as np
import pandas as pd

from training import build_high_volatility_spot_flow_leadership_relay_support as support


def frame():
    return pd.DataFrame(
        {
            "signal_valid": [True] * 4,
            "spot_late_return":[.015,-.015,.03,.015],"perp_late_return":[.01,-.01,-.01,.01],"spot_late_taker_imbalance":[.2,-.2,-.2,.2],"perp_late_taker_imbalance":[.1,-.1,-.1,.1],
            "variation_rank": [0.9, 0.8, 0.9, 0.4],
        }
    )


def test_primary_follows_flow_confirmed_late_direction():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False]
    assert side[active].tolist() == [1, -1]


def test_controls_remain_diagnostic_only():
    candidate = frame()
    assert support.conditions(candidate, "no_volatility_gate")[0].tolist() == [True, True, False, True]
    candidate.loc[1,"perp_late_taker_imbalance"]=-.15
    assert support.conditions(candidate,"no_leadership_ratio_gate")[0].tolist()==[True,True,False,False]
    active, side = support.conditions(frame(), "direction_fade")
    assert side[active].tolist() == [-1, 1]


def test_rank_excludes_current_observation():
    values = pd.Series(np.arange(181, dtype=float))
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[179] != ranks.iloc[179]
    assert ranks.iloc[180] == 1.0
