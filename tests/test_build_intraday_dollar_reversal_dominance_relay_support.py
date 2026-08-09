import numpy as np
import pandas as pd

from training import build_intraday_dollar_reversal_dominance_relay_support as support


def frame():
    return pd.DataFrame({
        "signal_valid": [True] * 5,
        "early_factor": [1.0, -1.0, 1.0, 1.0, -0.2],
        "late_factor": [-1.2, 1.5, 0.5, -0.5, 0.4],
        "raw_early_factor": [0.01, -0.01, 0.01, 0.01, -0.002],
        "raw_late_factor": [-0.012, 0.015, -0.005, -0.005, 0.004],
        "btc_realized_variation_rank": [0.7, 0.8, 0.9, 0.9, 0.4],
    })


def test_primary_requires_reversal_late_dominance_and_volatility():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False, False]
    assert side[active].tolist() == [1.0, -1.0]


def test_controls_are_diagnostic_only():
    candidate = frame()
    assert support.CONTROLS == ("no_volatility_gate", "no_reversal_gate", "no_late_dominance", "raw_return_factors", "one_session_stale_factor_geometry", "direction_flip", "forced_long")
    assert support.conditions(candidate, "no_volatility_gate")[0].tolist() == [True, True, False, False, True]
    assert support.conditions(candidate, "no_reversal_gate")[0].tolist() == [True, True, False, False, False]
    assert support.conditions(candidate, "no_late_dominance")[0].tolist() == [True, True, False, True, False]
    active, side = support.conditions(candidate, "direction_flip")
    assert side[active].tolist() == [-1.0, 1.0]
    assert support.conditions(candidate, "forced_long")[1].tolist() == [1.0] * 5


def test_causal_statistics_exclude_current_value():
    values = pd.Series(np.arange(61, dtype=float))
    zscore = support.causal_z(values, lookback=90, minimum=60)
    expected = (60 - values.iloc[:60].mean()) / values.iloc[:60].std(ddof=1)
    assert np.isclose(zscore.iloc[60], expected)
    assert support.strict_prior_midrank(values, lookback=90, minimum=60).iloc[60] == 1.0


def test_canonical_orientation_and_sealed_binding():
    assert support.DOLLAR_MULTIPLIER == {"EURUSD": -1.0, "GBPUSD": -1.0, "USDAUD": 1.0, "USDCAD": 1.0, "USDCHF": 1.0, "USDJPY": 1.0}
    assert support.sha(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    source = support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened":False' in source
    assert '"gross9_rows_opened":False' in source
