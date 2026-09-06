import numpy as np
import pandas as pd

from training import build_high_volatility_low_frequency_energy_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_low_frequency_share_is_high_for_slow_return_cycle():
    index = np.arange(479)
    returns = 0.001 * np.sin(2 * np.pi * 2 * index / 479)
    close = np.exp(np.r_[1.0, 1.0 + np.cumsum(returns)])
    assert support.low_frequency_share(close) > 0.99


def test_low_frequency_share_is_low_for_alternating_returns():
    returns = np.tile([0.01, -0.01], 240)[:479]
    close = np.exp(np.r_[1.0, 1.0 + np.cumsum(returns)])
    assert support.low_frequency_share(close) < 0.01


def test_contract_is_frozen():
    assert support.PREREG_SHA == "a4a6c0d2e0e6196aeda54536008372f322459a1aa147b947b3850973fac7e5cf"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_low_frequency_tail",
        "no_variation_gate",
        "raw_low_frequency_share_nonnegative",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
