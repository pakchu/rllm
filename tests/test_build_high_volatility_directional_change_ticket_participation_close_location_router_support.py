import numpy as np
import pandas as pd

from training import build_high_volatility_directional_change_ticket_participation_close_location_router_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_directional_change_count_is_zero_for_constant_log_returns():
    close = np.exp(np.linspace(1.0, 2.0, 480))
    count, threshold = support.directional_change_count(close)
    assert count == 0
    assert threshold > 0


def test_directional_change_count_is_large_for_alternating_log_returns():
    returns = np.tile([0.03, -0.03], 240)[:479]
    close = np.exp(np.r_[1.0, 1.0 + np.cumsum(returns)])
    count, threshold = support.directional_change_count(close)
    assert count > 400
    assert threshold > 0


def test_contract_is_frozen():
    assert support.PREREG_SHA == "b204e4e2d27d96ddc617d0223654801ca45001245696369baf9f5d238fb57bf7"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "always_follow_confirmed_direction",
        "always_fade_confirmed_direction",
        "no_joint_ticket_participation_acceleration_gate",
        "no_directional_change_scarcity_tail",
        "no_variation_gate",
        "raw_directional_changes_at_most_478",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
