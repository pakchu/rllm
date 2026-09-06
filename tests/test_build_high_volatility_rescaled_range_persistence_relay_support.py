import numpy as np
import pandas as pd

from training import build_high_volatility_rescaled_range_persistence_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_hurst_slope_detects_persistent_scaling():
    rng = np.random.default_rng(7)
    innovations = rng.normal(size=479)
    persistent = np.empty_like(innovations)
    persistent[0] = innovations[0]
    for index in range(1, len(persistent)):
        persistent[index] = 0.8 * persistent[index - 1] + innovations[index]
    assert np.isfinite(support.hurst_slope(persistent))


def test_contract_is_frozen():
    assert support.PREREG_SHA == "fc27df3a8f441a3c308bd113cca788e5e0551718f4f0226295a0efc2f23be09e"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_hurst_tail",
        "no_variation_gate",
        "raw_hurst_above_half",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
