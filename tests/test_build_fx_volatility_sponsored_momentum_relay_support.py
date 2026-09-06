import numpy as np
import pandas as pd

from training import build_fx_volatility_sponsored_momentum_relay_support as support


def frame():
    return pd.DataFrame(
        {
            "signal_valid": [True] * 5,
            "fx_shock": [1.0, 1.5, 0.5, 1.0, 0.2],
            "fx_shock_rank": [0.8, 0.9, 0.69, 0.8, 0.8],
            "raw_fx_absolute_return_shock": [0.01, 0.02, 0.005, 0.01, 0.002],
            "raw_fx_shock_rank": [0.9, 0.6, 0.8, 0.9, 0.8],
            "btc_session_return": [0.01, -0.02, -0.01, 0.02, 0.03],
            "btc_realized_variation_rank": [0.7, 0.8, 0.9, 0.4, 0.9],
        }
    )


def test_primary_trades_btc_momentum_when_fx_shock_sponsors_it():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False, True]
    assert side[active].tolist() == [1, -1, 1]


def test_controls_are_diagnostic_transformations():
    candidate = frame()
    assert support.CONTROLS == ("no_volatility_gate", "no_fx_shock_tail", "raw_fx_absolute_return_shock", "one_session_stale_fx_shock", "direction_flip")
    assert support.conditions(candidate, "no_volatility_gate")[0].tolist() == [True, True, False, True, True]
    assert support.conditions(candidate, "no_fx_shock_tail")[0].tolist() == [True, True, True, False, True]
    raw_active, raw_side = support.conditions(candidate, "raw_fx_absolute_return_shock")
    assert raw_active.tolist() == [True, False, True, False, True]
    assert raw_side[raw_active].tolist() == [1, -1, 1]
    active, side = support.conditions(frame(), "direction_flip")
    assert side[active].tolist() == [-1, 1, -1]


def test_causal_statistics_exclude_current_value():
    values = pd.Series(np.arange(61, dtype=float))
    zscore = support.causal_z(values, lookback=90, minimum=60)
    expected = (60 - values.iloc[:60].mean()) / values.iloc[:60].std(ddof=1)
    assert np.isclose(zscore.iloc[60], expected)
    rank = support.strict_prior_midrank(values, lookback=90, minimum=60)
    assert rank.iloc[60] == 1.0


def test_fx_direction_is_not_canonicalized_or_used_for_side():
    assert not hasattr(support, "DOLLAR_MULTIPLIER")
    source = support.Path(support.__file__).read_text()
    assert '"fx_direction_used":False' in source


def test_builder_binds_preregistration_and_seals_outcomes():
    assert support.sha(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    source = support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened":False' in source
    assert '"gross9_rows_opened":False' in source
