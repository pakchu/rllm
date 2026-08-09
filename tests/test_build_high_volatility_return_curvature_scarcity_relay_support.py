import numpy as np
import pandas as pd

from training import build_high_volatility_return_curvature_scarcity_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_return_curvature_is_zero_for_constant_log_returns():
    close = np.exp(np.linspace(1.0, 2.0, 480))
    assert support.return_curvature_ratio(close) < 1e-12


def test_return_curvature_is_large_for_alternating_log_returns():
    returns = np.tile([0.01, -0.01], 240)[:479]
    close = np.exp(np.r_[1.0, 1.0 + np.cumsum(returns)])
    assert support.return_curvature_ratio(close) > 0.99


def test_contract_is_frozen():
    assert support.PREREG_SHA == "306172f8379ee38236634cec6482d7b6d2be837c1451990a99f81c7cb1e331f7"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_curvature_scarcity_tail",
        "no_variation_gate",
        "raw_curvature_ratio_at_most_half",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
