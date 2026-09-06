import numpy as np
import pandas as pd

from training import build_high_volatility_weekly_momentum_relay_support as support


def frame():
    return pd.DataFrame({
        "decision_time": pd.date_range("2024-01-03T00:00:00Z", periods=4, freq="7D"),
        "source_valid": [True] * 4,
        "weekly_return": [0.1, -0.1, 0.1, -0.1],
        "weekly_realized_variation": [0.2] * 4,
        "variation_rank": [0.7, 0.5, 0.8, 0.9],
    })


def test_rank_excludes_current_week():
    ranked = support.strict_prior_midrank(pd.Series([1., 2., 3.]), lookback=2, minimum=2)
    assert np.isnan(ranked.iloc[:2]).all() and ranked.iloc[2] == 1.


def test_primary_and_controls_are_frozen():
    values = frame(); active, side = support.conditions(values, "primary")
    assert active.tolist() == [True, False, True, True]
    assert side.tolist() == [1., -1., 1., -1.]
    assert support.conditions(values, "no_volatility_gate")[0].tolist() == [True] * 4
    assert support.conditions(values, "direction_flip")[1].tolist() == [-1., 1., -1., 1.]
    assert support.conditions(values, "forced_long")[1].tolist() == [1.] * 4


def test_clock_delays_five_minutes_and_holds_72h():
    clock = support.build_clock(frame())
    assert len(clock) == 3
    assert clock.iloc[0].entry_time == pd.Timestamp("2024-01-03T00:05:00Z")
    assert clock.iloc[0].exit_time == pd.Timestamp("2024-01-06T00:05:00Z")


def test_preregistration_binding_and_sealed_outcomes():
    assert support.sha256(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    source = support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
