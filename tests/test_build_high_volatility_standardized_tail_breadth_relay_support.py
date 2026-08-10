import numpy as np
import pandas as pd

from training import build_high_volatility_standardized_tail_breadth_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_tail_statistics_identifies_positive_standardized_breadth():
    returns = np.r_[np.full(30, 0.03), np.full(10, -0.03), np.zeros(439)]
    close = np.exp(np.r_[1.0, 1.0 + np.cumsum(returns)])
    rms, positive, negative, breadth, share = support.tail_statistics(close)
    assert rms > 0
    assert positive == 30
    assert negative == 10
    assert breadth == 20
    assert share == 0.5


def test_tail_statistics_rejects_constant_close():
    rms, positive, negative, breadth, share = support.tail_statistics(np.ones(480))
    assert np.isnan(rms)
    assert (positive, negative, breadth) == (0, 0, 0)
    assert np.isnan(share)


def test_contract_is_frozen():
    assert support.PREREG_SHA == "6d70486179ac1ac8b35c53fbff02a580e526207216f286c746ef7659c45b69ca"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_tail_breadth_share_gate",
        "no_variation_gate",
        "raw_absolute_tail_breadth_at_least_one",
        "one_block_stale_features",
        "direction_flip",
        "forced_long",
    )
