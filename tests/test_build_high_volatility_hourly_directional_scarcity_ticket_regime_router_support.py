import numpy as np
import pandas as pd

from training import build_high_volatility_hourly_directional_scarcity_ticket_regime_router_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(1441)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[1439])
    assert ranks.iloc[1440] == 1.0


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
    assert support.PREREG_SHA == "f1ac9cff4cdbca9f360ae71ac673eb5ca880303336c42e0d97662a9c732c4fcc"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "always_follow_confirmed_direction",
        "always_fade_confirmed_direction",
        "no_trade_participation_gate",
        "no_directional_change_scarcity_tail",
        "no_variation_gate",
        "one_hour_stale_geometry",
        "direction_flip",
        "forced_long",
    )


def test_ticket_router_follows_large_ticket_and_fades_small_ticket():
    states = pd.DataFrame(
        {
            "source_valid": [True, True, True],
            "block_return": [0.01, 0.01, 0.01],
            "late_return": [0.005, 0.005, 0.005],
            "variation_rank": [0.7, 0.7, 0.7],
            "directional_change_rank": [0.2, 0.2, 0.2],
            "directional_change_count": [2.0, 2.0, 2.0],
            "average_ticket_acceleration": [0.1, 0.1, -0.1],
            "trade_count_acceleration": [-0.1, 0.1, 0.1],
        }
    )
    onset, side = support.active(states, "primary")
    assert onset.tolist() == [False, True, False]
    assert side.tolist() == [1.0, 1.0, -1.0]
