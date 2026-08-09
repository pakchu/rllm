import numpy as np
import pandas as pd

from training import build_high_volatility_temporal_rank_concordance_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_temporal_rank_concordance_tracks_monotone_direction():
    increasing = np.exp(np.linspace(1.0, 2.0, 480))
    decreasing = increasing[::-1]
    assert support.temporal_rank_concordance(increasing) > 0.999999
    assert support.temporal_rank_concordance(decreasing) < -0.999999


def test_contract_is_frozen():
    assert support.PREREG_SHA == "87556ccab968c68b7ec70967680eaa6e57057e34960fb0b6e4db1aa29fbc0f39"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_concordance_tail",
        "no_variation_gate",
        "raw_absolute_concordance_positive",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
