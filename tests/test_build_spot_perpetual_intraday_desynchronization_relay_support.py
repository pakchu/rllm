import numpy as np
import pandas as pd

from training import build_spot_perpetual_intraday_desynchronization_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_return_correlation_detects_synchronization():
    first = np.linspace(-1.0, 1.0, 480)
    assert np.isclose(support.return_correlation(first, first), 1.0)
    assert np.isclose(support.return_correlation(first, -first), -1.0)


def test_contract_is_frozen():
    assert support.PREREG_SHA == "9314697e95fc1e8a1210ca9574631b71644c445a7d8ab638be8d7e0389cdb5cb"
    assert "FROM bars_binance" in support.PERPETUAL_QUERY
    assert "FROM bars_binance_spot" in support.SPOT_QUERY
    assert support.CONTROLS == (
        "no_desynchronization_gate",
        "no_variation_gate",
        "raw_correlation_below_half",
        "one_block_stale_desynchronization",
        "direction_flip",
        "forced_long",
    )
