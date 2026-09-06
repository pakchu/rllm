import numpy as np
import pandas as pd

from training import build_high_volatility_shock_response_polarity_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_response_correlation_is_finite():
    rng = np.random.default_rng(7)
    returns = rng.normal(size=480)
    assert np.isfinite(support.response_correlation(returns))
    synthetic = np.empty(480)
    synthetic[:-1] = np.linspace(-2.0, 2.0, 479)
    synthetic[1:] = np.abs(synthetic[:-1])
    assert support.response_correlation(synthetic) > 0


def test_contract_is_frozen():
    assert support.PREREG_SHA == "2164103502c9bebabce4ece727e961533076e8b1e05bf1b145f7f0765c06e09e"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_response_tail",
        "no_variation_gate",
        "raw_absolute_response_above_tenth",
        "one_block_stale_response",
        "direction_flip",
        "forced_long",
    )
