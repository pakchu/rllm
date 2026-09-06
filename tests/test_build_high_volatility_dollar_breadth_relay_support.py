import numpy as np
import pandas as pd

from training import build_high_volatility_dollar_breadth_relay_support as support


def frame():
    return pd.DataFrame(
        {
            "signal_valid": [True] * 5,
            "common_dollar_direction": [1, -1, 1, 1, -1],
            "agreeing_pairs": [5, 6, 4, 5, 5],
            "median_absolute_pair_z": [1.2, 1.5, 1.4, 1.3, 0.7],
            "btc_realized_variation_rank": [0.7, 0.8, 0.9, 0.4, 0.9],
        }
    )


def test_primary_trades_opposite_broad_dollar_shock():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False, False]
    assert side[active].tolist() == [-1, 1]


def test_controls_are_diagnostic_transformations():
    candidate = frame()
    assert support.conditions(candidate, "no_volatility_gate")[0].tolist() == [True, True, False, True, False]
    assert support.conditions(candidate, "no_dollar_shock_gate")[0].tolist() == [True, True, False, False, True]
    assert support.conditions(candidate, "no_breadth_gate")[0].tolist() == [True, True, True, False, False]
    active, side = support.conditions(frame(), "direction_flip")
    assert side[active].tolist() == [1, -1]


def test_causal_statistics_exclude_current_value():
    values = pd.Series(np.arange(61, dtype=float))
    zscore = support.causal_z(values, lookback=90, minimum=60)
    expected = (60 - values.iloc[:60].mean()) / values.iloc[:60].std(ddof=1)
    assert np.isclose(zscore.iloc[60], expected)
    rank = support.strict_prior_midrank(values, lookback=90, minimum=60)
    assert rank.iloc[60] == 1.0


def test_canonical_dollar_orientation_is_frozen():
    assert support.DOLLAR_MULTIPLIER == {"EURUSD": -1.0, "GBPUSD": -1.0, "USDAUD": 1.0, "USDCAD": 1.0, "USDCHF": 1.0, "USDJPY": 1.0}
