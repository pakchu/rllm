import numpy as np
import pandas as pd

from training import build_high_volatility_adverse_underwater_duration_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_adverse_duration_distinguishes_clean_trend():
    rng = np.random.default_rng(7)
    innovations = rng.normal(size=479)
    persistent = np.empty_like(innovations)
    persistent[0] = innovations[0]
    for index in range(1, len(persistent)):
        persistent[index] = 0.8 * persistent[index - 1] + innovations[index]
    assert support.adverse_duration(np.arange(1.0, 481.0), 1) == 0.0
    assert support.adverse_duration(np.arange(480.0, 0.0, -1.0), -1) == 0.0


def test_contract_is_frozen():
    assert support.PREREG_SHA == "5a8d279c9b6900482ed9f43ba8054aa368498d4bad7a96af4acffc2bb0ee8286"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_duration_tail",
        "no_variation_gate",
        "raw_duration_at_most_half",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
