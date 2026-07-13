from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_spot_perp_absorption_alpha import (
    _make_event,
    _prior_z,
    _rolling_residual,
    _signals,
)


def test_prior_z_does_not_fit_on_current_value() -> None:
    values = pd.Series(([-1.0, 1.0] * 150) + [100.0])
    z = _prior_z(values, 288)
    assert z.iloc[-1] > 90.0


def test_rolling_residual_beta_uses_only_prior_rows() -> None:
    x = pd.Series(np.arange(400, dtype=float))
    y = 2.0 * x
    residual, beta = _rolling_residual(y, x, 288)
    baseline_beta = beta.iloc[-1]
    y.iloc[-1] = 1_000_000.0
    changed_residual, changed_beta = _rolling_residual(y, x, 288)
    assert changed_beta.iloc[-1] == baseline_beta
    assert changed_residual.iloc[-1] != residual.iloc[-1]


def test_signal_is_contra_residual_after_contraction() -> None:
    frame = pd.DataFrame(
        {
            "spa_residual_z_2016": [np.nan, 3.2, 2.8],
            "spa_lead_residual_z_2016": [0.0, 0.0, -1.0],
            "spa_flow_z_2016": [0.0, 0.0, 0.0],
            "spa_spot_share_z_2016": [0.0, 0.0, 1.0],
        }
    )
    spec = {"window": 2016, "z_entry": 3.0, "contraction_delta": 0.25, "mode": "spot_absorption", "direction": "contra", "phase": "contraction", "max_hold": 48}
    active, side = _signals(frame, spec)
    assert active.tolist() == [False, False, True]
    assert side.tolist() == [0, 0, -1]


def test_expansion_onset_fires_only_on_first_threshold_cross() -> None:
    frame = pd.DataFrame(
        {
            "spa_residual_z_2016": [1.0, 3.2, 3.4],
            "spa_lead_residual_z_2016": [0.0, 1.0, 1.0],
            "spa_flow_z_2016": [0.0, 0.0, 0.0],
            "spa_spot_share_z_2016": [0.0, 0.0, 0.0],
        }
    )
    spec = {"window": 2016, "z_entry": 3.0, "contraction_delta": 0.0, "mode": "lead_residual", "direction": "continuation", "phase": "expansion_onset", "max_hold": 48}
    active, side = _signals(frame, spec)
    assert active.tolist() == [False, True, False]
    assert side.tolist() == [0, 1, 0]


def test_dynamic_exit_occurs_next_open_after_cross() -> None:
    market = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0] * 6,
            "low": [99.0] * 6,
        }
    )
    residual_z = np.asarray([3.0, 2.0, 0.4, 0.2, 0.0, 0.0])
    event = _make_event(
        market,
        residual_z,
        signal_pos=0,
        side=-1,
        residual_sign=1,
        max_hold=4,
        dynamic_exit=True,
        exit_abs_z=0.5,
        cost_rate=0.0006,
        leverage=0.5,
        name="test",
    )
    assert event is not None
    assert event["entry_pos"] == 1
    assert event["exit_pos"] == 3
