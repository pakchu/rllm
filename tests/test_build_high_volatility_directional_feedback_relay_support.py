import numpy as np
import pandas as pd

from training import build_high_volatility_directional_feedback_relay_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_feedback_correlation_is_finite():
    rng = np.random.default_rng(7)
    returns = rng.normal(size=480)
    assert np.isfinite(support.feedback_correlation(returns))


def test_contract_is_frozen():
    assert support.PREREG_SHA == "111da91ab5e052ebb9eb10288ea94fde13e926ad2158eeaaf07d4a4a197bb361"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_feedback_tail",
        "no_variation_gate",
        "raw_absolute_feedback_above_tenth",
        "one_block_stale_feedback",
        "direction_flip",
        "forced_long",
    )
