import numpy as np
import pandas as pd

from training import build_high_volatility_lempel_ziv_compressibility_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_lz_complexity_is_lower_for_repeated_sequence():
    rng = np.random.default_rng(7)
    innovations = rng.normal(size=479)
    persistent = np.empty_like(innovations)
    persistent[0] = innovations[0]
    for index in range(1, len(persistent)):
        persistent[index] = 0.8 * persistent[index - 1] + innovations[index]
    random = rng.choice([-1.0, 1.0], size=479)
    repeated = np.resize(np.array([1.0, 1.0, -1.0]), 479)
    assert support.normalized_complexity(repeated) < support.normalized_complexity(random)


def test_contract_is_frozen():
    assert support.PREREG_SHA == "1aff82bc58760b8dfc8c14798c20085dc8e9387caaec71f478564855e6afda8b"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_complexity_tail",
        "no_variation_gate",
        "raw_below_unit_complexity",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
