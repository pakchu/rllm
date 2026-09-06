import numpy as np
import pandas as pd

from training import build_dollar_factor_dual_phase_relay_support as support


def frame():
    return pd.DataFrame({
        "signal_valid": [True] * 6, "source_valid": [True] * 6, "persistent": [True, True, False, True, True, True],
        "btc_valid": [True] * 6, "early_factor": [1., -1., 1., 1., -1., 1.],
        "late_factor": [.5, -.5, -.5, .5, -.5, .5],
        "early_absolute_rank": [.8] * 6, "late_absolute_rank": [.8] * 6,
        "persistence_rank": [.8, .9, .9, .5, .8, .9],
        "btc_realized_variation_rank": [.7, .8, .9, .9, .4, .9],
    })


def test_causal_statistics_exclude_current():
    values = pd.Series(np.arange(61, dtype=float))
    zscore = support.causal_z(values)
    expected = (60 - values.iloc[:60].mean()) / values.iloc[:60].std(ddof=1)
    assert np.isclose(zscore.iloc[60], expected)
    assert support.strict_prior_midrank(values).iloc[60] == 1.0


def test_primary_requires_persistent_phases_rank_and_volatility():
    active, side = support.conditions(frame())
    assert active.tolist() == [True, True, False, False, False, True]
    assert side[active].tolist() == [-1, 1, -1]


def test_controls_are_frozen_before_incidence():
    data = frame()
    assert support.CONTROLS == (
        "no_persistence_rank", "no_variation_gate", "early_factor_only", "late_factor_only",
        "one_session_stale_phases", "direction_flip", "same_clock_forced_long",
    )
    assert support.conditions(data, "no_persistence_rank")[0].tolist() == [True, True, False, True, False, True]
    assert support.conditions(data, "no_variation_gate")[0].tolist() == [True, True, False, False, True, True]
    active, side = support.conditions(data, "direction_flip")
    assert side[active].tolist() == [1, -1, 1]
    active, side = support.conditions(data, "same_clock_forced_long")
    assert side[active].tolist() == [1, 1, 1]


def test_phase_orientation_and_preregistration_binding_are_frozen():
    assert support.DOLLAR_MULTIPLIER == {
        "EURUSD": -1.0, "GBPUSD": -1.0, "USDAUD": 1.0,
        "USDCAD": 1.0, "USDCHF": 1.0, "USDJPY": 1.0,
    }
    assert support.sha(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    source = support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
