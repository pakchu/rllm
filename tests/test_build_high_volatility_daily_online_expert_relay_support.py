import numpy as np
import pandas as pd

from training import build_high_volatility_daily_online_expert_relay_support as support


def states():
    return pd.DataFrame({
        "signal_valid": [True] * 5, "winner_side": [1, -1, 1, -1, 1],
        "momentum_6h_side": [-1, 1, -1, 1, -1], "momentum_24h_side": [1, 1, -1, -1, 1],
        "variation_rank": [.7, .8, .5, .9, .9],
    })


def test_expert_side_formulas_and_tie_order_are_frozen():
    assert support.EXPERTS == ("momentum_6h", "reversal_6h", "momentum_24h", "reversal_24h")
    assert support.expert_sides(0.1, -0.2).tolist() == [1, -1, -1, 1]


def test_primary_and_controls_use_frozen_clock_eligibility():
    active, side = support.conditions(states())
    assert active.tolist() == [True, True, False, True, True]
    assert side[active].tolist() == [1, -1, -1, 1]
    assert support.conditions(states(), "no_variation_gate")[0].tolist() == [True] * 5
    assert support.conditions(states(), "fixed_momentum_6h")[1].tolist() == [-1, 1, -1, 1, -1]
    assert support.conditions(states(), "direction_flip")[1].tolist() == [-1, 1, -1, 1, -1]
    assert support.conditions(states(), "same_clock_forced_long")[1].tolist() == [1] * 5


def test_rank_excludes_current_and_preregistration_is_bound():
    values = pd.Series(np.arange(127, dtype=float))
    rank = support.strict_prior_midrank(values)
    assert np.isnan(rank.iloc[125]) and rank.iloc[126] == 1.0
    assert support.sha(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    source = support.Path(support.__file__).read_text()
    assert '"same_decision_label_used": False' in source
    assert '"unmatured_label_used": False' in source
