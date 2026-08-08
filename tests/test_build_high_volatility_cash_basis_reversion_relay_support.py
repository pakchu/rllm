import numpy as np
import pandas as pd

from training import build_high_volatility_cash_basis_reversion_relay_support as support


def frame():
    return pd.DataFrame(
        {
            "signal_valid": [True] * 4,
            "basis_change":[-.001,.001,0.,-.001],"absolute_basis_change_rank":[.9,.8,.9,.9],
            "variation_rank": [0.9, 0.8, 0.9, 0.4],
        }
    )


def test_primary_fades_extreme_basis_change():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False]
    assert side[active].tolist() == [1, -1]


def test_controls_remain_diagnostic_only():
    candidate = frame()
    assert support.conditions(candidate, "no_volatility_gate")[0].tolist() == [True, True, False, True]
    candidate.loc[1,"absolute_basis_change_rank"]=.5
    assert support.conditions(candidate,"no_basis_shock_gate")[0].tolist()==[True,True,False,False]
    active, side = support.conditions(frame(), "direction_follow_basis")
    assert side[active].tolist() == [-1, 1]


def test_rank_excludes_current_observation():
    values = pd.Series(np.arange(181, dtype=float))
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[179] != ranks.iloc[179]
    assert ranks.iloc[180] == 1.0
