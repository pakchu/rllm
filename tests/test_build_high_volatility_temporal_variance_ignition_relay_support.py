import numpy as np
import pandas as pd

from training import build_high_volatility_temporal_variance_ignition_relay_support as support


def frame():
    return pd.DataFrame(
        {
            "decision_time": pd.date_range("2024-01-01T02:00:00Z", periods=4, freq="1D"),
            "source_valid": [True] * 4,
            "early_return": [0.02, -0.02, 0.02, 0.02],
            "late_return": [0.01, -0.01, -0.01, 0.01],
            "direction_agreement": [True, True, False, True],
            "realized_variation": [0.1] * 4,
            "late_variance_share": [0.3, 0.3, 0.3, 0.1],
            "variation_rank": [0.8, 0.5, 0.8, 0.8],
            "ignition_rank": [0.9, 0.9, 0.9, 0.7],
        }
    )


def test_rank_is_strict_prior():
    ranked = support.strict_prior_midrank(pd.Series([1.0, 2.0, 3.0]), lookback=2, minimum=2)
    assert np.isnan(ranked.iloc[:2]).all()
    assert ranked.iloc[2] == 1.0


def test_primary_and_controls_are_frozen():
    values = frame()
    active, side = support.conditions(values, "primary")
    assert active.tolist() == [True, False, False, False]
    assert side.tolist() == [1.0, -1.0, -1.0, 1.0]
    assert support.conditions(values, "no_volatility_gate")[0].tolist() == [True, True, False, False]
    assert support.conditions(values, "no_ignition_tail")[0].tolist() == [True, False, False, True]
    assert support.conditions(values, "no_direction_agreement")[0].tolist() == [True, False, True, False]
    assert support.conditions(values, "direction_flip")[1].tolist() == [-1.0, 1.0, 1.0, -1.0]
    assert support.conditions(values, "forced_long")[1].tolist() == [1.0] * 4


def test_clock_delays_five_minutes_and_holds_twelve_hours():
    clock = support.build_clock(frame())
    assert len(clock) == 1
    assert clock.iloc[0].entry_time == pd.Timestamp("2024-01-01T02:05:00Z")
    assert clock.iloc[0].exit_time == pd.Timestamp("2024-01-01T14:05:00Z")


def test_preregistration_binding_and_sealed_outcomes():
    assert support.sha256(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    source = support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
