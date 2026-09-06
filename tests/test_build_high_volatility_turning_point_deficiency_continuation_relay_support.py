import numpy as np
import pandas as pd

from training import build_high_volatility_turning_point_deficiency_continuation_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_turning_point_count_is_zero_for_monotone_path():
    close = np.exp(np.linspace(1.0, 2.0, 96))
    assert support.turning_point_count(close) == 0


def test_turning_point_count_is_maximal_for_alternating_path():
    close = np.tile([1.0, 2.0], 48)
    assert support.turning_point_count(close) == 94


def test_turning_point_count_ignores_equal_neighbors():
    close = np.arange(96, dtype=float) + 1.0
    close[20:23] = close[20]
    assert support.turning_point_count(close) == 0


def test_contract_is_frozen():
    assert support.PREREG_SHA == "57ffeebf061e7adf8e7517e1c9287f94ba8a3360a5e91fc13bb35f12c12d3ae1"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_turning_point_deficiency_tail",
        "no_variation_gate",
        "turning_point_share_below_two_thirds",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
