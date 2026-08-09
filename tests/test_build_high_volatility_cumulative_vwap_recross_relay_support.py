import numpy as np
import pandas as pd

from training import build_high_volatility_cumulative_vwap_recross_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_recross_count_detects_anchor_changes():
    rng = np.random.default_rng(7)
    innovations = rng.normal(size=479)
    persistent = np.empty_like(innovations)
    persistent[0] = innovations[0]
    for index in range(1, len(persistent)):
        persistent[index] = 0.8 * persistent[index - 1] + innovations[index]
    assert np.isfinite(support.recross_count(np.linspace(1.0, 2.0, 480), np.ones(480), np.linspace(0.9, 1.9, 480)))


def test_contract_is_frozen():
    assert support.PREREG_SHA == "9211f953469484a3788660bb5e7746707ee610747f9664a930243574e5a7dedb"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_recross_tail",
        "no_variation_gate",
        "raw_recross_at_most_eight",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
