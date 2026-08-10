import numpy as np
import pandas as pd

from training import build_high_volatility_intraday_distribution_break_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_ks_distance_detects_distribution_break():
    first = np.linspace(-1.0, 1.0, 240)
    same = first.copy()
    shifted = first + 2.0
    assert support.ks_distance(first, same) == 0.0
    assert support.ks_distance(first, shifted) > 0.5


def test_contract_is_frozen():
    assert support.PREREG_SHA == "a8592e5b817e97dfb0687a1d131ef61114cf67517d708793ad96aa0718092efd"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_distribution_break_gate",
        "no_variation_gate",
        "raw_ks_above_quarter",
        "one_block_stale_distribution",
        "direction_flip",
        "forced_long",
    )
