import numpy as np
import pandas as pd

from training import build_high_volatility_btc_dominance_rotation_relay_support as support


def frame():
    return pd.DataFrame(
        {
            "session_date": ["2024-01-01"] * 5,
            "decision_time": pd.date_range("2024-01-01T04:00:00Z", periods=5, freq="8h"),
            "source_valid": [True] * 5,
            "btc_return": [0.03, -0.03, 0.02, 0.01, -0.02],
            "alt_factor": [0.01, -0.01, 0.03, -0.01, 0.01],
            "btc_dominance_residual": [0.02, -0.02, -0.01, 0.02, -0.03],
            "absolute_residual_rank": [0.8, 0.8, 0.6, 0.8, 0.9],
            "alt_dispersion": [0.01] * 5,
            "alt_dispersion_rank": [0.8, 0.5, 0.8, 0.8, 0.9],
            "btc_realized_variation": [0.1] * 5,
            "variation_rank": [0.8, 0.8, 0.8, 0.5, 0.9],
        }
    )


def test_rank_excludes_current_and_caps_history():
    rank = support.strict_prior_midrank(pd.Series([1.0, 2.0, 3.0, 2.0]), 2, 2)
    assert np.isnan(rank.iloc[:2]).all()
    assert rank.iloc[2] == 1.0
    assert rank.iloc[3] == 0.25


def test_primary_and_frozen_controls():
    data = frame()
    active, side, _ = support.conditions(data)
    assert active.tolist() == [True, False, False, False, True]
    assert side.tolist() == [1, -1, -1, 1, -1]
    assert support.conditions(data, "no_residual_tail")[0].tolist() == [True, False, True, False, True]
    assert support.conditions(data, "no_dispersion_gate")[0].tolist() == [True, True, False, False, True]
    assert support.conditions(data, "no_variation_gate")[0].tolist() == [True, False, False, True, True]
    assert support.conditions(data, "direction_flip")[1].tolist() == [-1, 1, 1, -1, 1]
    assert support.conditions(data, "same_clock_forced_long")[1].tolist() == [1] * 5


def test_alt_factor_control_keeps_clock_and_changes_only_side():
    primary_active, _, _ = support.conditions(frame())
    active, side, _ = support.conditions(frame(), "alt_factor_direction")
    assert active.equals(primary_active)
    assert side[active].tolist() == [1, 1]


def test_clock_delay_hold_and_binding_are_frozen():
    candidate = support.clock(frame())
    assert candidate.iloc[0].entry_time == pd.Timestamp("2024-01-01T04:05:00Z")
    assert candidate.iloc[0].exit_time == pd.Timestamp("2024-01-01T12:05:00Z")
    assert support.sha(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    source = support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
