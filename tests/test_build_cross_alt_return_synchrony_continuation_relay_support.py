import numpy as np
import pandas as pd

from training import build_cross_alt_return_synchrony_continuation_relay_support as support


def test_prior_rank_excludes_current():
    values = pd.Series(np.arange(181), dtype=float)
    ranks = support.strict_prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_primary_requires_onset_and_uses_btc_side():
    rows = 5
    frame = pd.DataFrame(
        {
            "source_valid": [True] * rows,
            "synchrony": [0.8] * rows,
            "synchrony_rank": [0.1, 0.8, 0.9, 0.1, 0.9],
            "btc_return": [0.01] * rows,
            "alt_breadth": [6] * rows,
            "btc_variation_rank": [0.8] * rows,
        }
    )
    active, side = support.conditions(frame, "primary")
    assert active.tolist() == [False, True, False, False, True]
    assert side.tolist() == [1.0] * rows


def test_contract_is_hash_bound():
    assert support.PREREG_SHA == "fa9ed9a30e8cbd0532b690acd585d371c91c91a725e7c1b8f4f3187544d0c8e4"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_synchrony_gate",
        "raw_median_correlation_above_half",
        "one_block_stale_synchrony",
        "direction_flip",
        "forced_long",
    )
