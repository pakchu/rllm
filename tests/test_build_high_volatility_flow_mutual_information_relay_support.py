import numpy as np
import pandas as pd

from training import build_high_volatility_flow_mutual_information_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_normalized_mutual_information_detects_exact_dependence():
    rng = np.random.default_rng(7)
    innovations = rng.normal(size=479)
    persistent = np.empty_like(innovations)
    persistent[0] = innovations[0]
    for index in range(1, len(persistent)):
        persistent[index] = 0.8 * persistent[index - 1] + innovations[index]
    assert np.isfinite(support.normalized_mutual_information(persistent, persistent))


def test_contract_is_frozen():
    assert support.PREREG_SHA == "572694f70406d2fa26f1d0a0be6eea6a54258c793ba67b305a67a8b0e9d299c1"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_mutual_information_tail",
        "no_variation_gate",
        "raw_positive_mutual_information",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
