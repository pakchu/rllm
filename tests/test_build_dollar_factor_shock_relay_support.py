import numpy as np
import pandas as pd

from training import build_dollar_factor_shock_relay_support as support


def frame():
    return pd.DataFrame(
        {
            "signal_valid": [True] * 5,
            "dollar_factor": [1.0, -1.0, 0.5, 1.0, -0.2],
            "absolute_factor_rank": [0.8, 0.9, 0.69, 0.8, 0.8],
            "raw_return_factor": [1.0, -1.0, -0.5, 1.0, 0.2],
            "btc_realized_variation_rank": [0.7, 0.8, 0.9, 0.4, 0.9],
        }
    )


def test_primary_trades_opposite_extreme_dollar_factor():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False, True]
    assert side[active].tolist() == [-1, 1, 1]


def test_controls_are_diagnostic_transformations():
    candidate = frame()
    assert support.CONTROLS == ("no_volatility_gate", "no_factor_tail", "raw_return_factor", "one_session_stale_factor", "direction_flip")
    assert support.conditions(candidate, "no_volatility_gate")[0].tolist() == [True, True, False, True, True]
    assert support.conditions(candidate, "no_factor_tail")[0].tolist() == [True, True, True, False, True]
    raw_active, raw_side = support.conditions(candidate, "raw_return_factor")
    assert raw_active.tolist() == [True, True, False, False, True]
    assert raw_side[raw_active].tolist() == [-1, 1, -1]
    active, side = support.conditions(frame(), "direction_flip")
    assert side[active].tolist() == [1, -1, -1]


def test_causal_statistics_exclude_current_value():
    values = pd.Series(np.arange(61, dtype=float))
    zscore = support.causal_z(values, lookback=90, minimum=60)
    expected = (60 - values.iloc[:60].mean()) / values.iloc[:60].std(ddof=1)
    assert np.isclose(zscore.iloc[60], expected)
    rank = support.strict_prior_midrank(values, lookback=90, minimum=60)
    assert rank.iloc[60] == 1.0


def test_canonical_dollar_orientation_is_frozen():
    assert support.DOLLAR_MULTIPLIER == {"EURUSD": -1.0, "GBPUSD": -1.0, "USDAUD": 1.0, "USDCAD": 1.0, "USDCHF": 1.0, "USDJPY": 1.0}


def test_builder_binds_preregistration_and_seals_outcomes():
    assert support.sha(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    source = support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened":False' in source
    assert '"gross9_rows_opened":False' in source
