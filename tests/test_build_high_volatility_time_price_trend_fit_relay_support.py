import numpy as np
import pandas as pd

from training import build_high_volatility_time_price_trend_fit_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_trend_fit_is_exact_for_log_linear_prices():
    rng = np.random.default_rng(7)
    innovations = rng.normal(size=479)
    persistent = np.empty_like(innovations)
    persistent[0] = innovations[0]
    for index in range(1, len(persistent)):
        persistent[index] = 0.8 * persistent[index - 1] + innovations[index]
    assert support.trend_fit(np.exp(np.linspace(1.0, 2.0, 480)))[1] > 0.999999


def test_contract_is_frozen():
    assert support.PREREG_SHA == "2bbf372142c15f00714bfc564013a4de4e326f291e447491d8c3d28519d67745"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_trend_fit_tail",
        "no_variation_gate",
        "raw_r_squared_at_least_half",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
