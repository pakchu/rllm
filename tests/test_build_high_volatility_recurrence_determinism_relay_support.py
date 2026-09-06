import numpy as np
import pandas as pd

from training import build_high_volatility_recurrence_determinism_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_recurrence_determinism_is_bounded():
    rng = np.random.default_rng(7)
    innovations = rng.normal(size=479)
    persistent = np.empty_like(innovations)
    persistent[0] = innovations[0]
    for index in range(1, len(persistent)):
        persistent[index] = 0.8 * persistent[index - 1] + innovations[index]
    value = support.recurrence_determinism(persistent)
    assert 0.0 <= value <= 1.0


def test_contract_is_frozen():
    assert support.PREREG_SHA == "fb1ebcf09a0b9fb12861e6ed03ae8706b131773cd0a69905ca647200e7787339"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_determinism_tail",
        "no_variation_gate",
        "raw_determinism_at_least_half",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
