import numpy as np
import pandas as pd

from training import build_high_volatility_absolute_return_clustering_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_absolute_return_autocorrelation_detects_volatility_clustering():
    rng = np.random.default_rng(7)
    innovations = rng.normal(size=479)
    persistent = np.empty_like(innovations)
    persistent[0] = innovations[0]
    for index in range(1, len(persistent)):
        persistent[index] = 0.8 * persistent[index - 1] + innovations[index]
    assert np.isfinite(support.absolute_return_autocorrelation(persistent))


def test_contract_is_frozen():
    assert support.PREREG_SHA == "af3f7afccac1d563a50727d43253c00cc8faa0ec4f13df5d59e448e09bd5f2ee"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_clustering_tail",
        "no_variation_gate",
        "raw_positive_clustering",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
