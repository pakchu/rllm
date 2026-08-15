import numpy as np
import pandas as pd

from training import build_high_volatility_directional_change_scarcity_open_interest_sponsorship_relay_support as support


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
    assert support.PREREG_SHA == "098c6e713e8db00eef987911170783b81eaac7ffa34b2e4bbdcacd254e16152a"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_open_interest_sponsorship_gate",
        "no_directional_change_scarcity_tail",
        "no_variation_gate",
        "raw_directional_changes_at_most_478",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )


def test_primary_requires_strict_open_interest_expansion():
    states = pd.DataFrame(
        {
            "source_valid": [True, True, True],
            "block_return": [0.1, 0.1, 0.1],
            "late_return": [0.1, 0.1, 0.1],
            "variation_rank": [0.8, 0.8, 0.8],
            "directional_change_rank": [0.1, 0.1, 0.1],
            "directional_change_count": [1, 1, 1],
            "oi_change": [-0.1, 0.1, -0.1],
        }
    )
    primary, side = support.active(states, "primary")
    assert primary.tolist() == [False, True, False]
    assert side[primary].tolist() == [1.0]
    no_oi, _ = support.active(states, "no_open_interest_sponsorship_gate")
    assert no_oi.tolist() == [False, False, False]
