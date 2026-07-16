from __future__ import annotations

import pandas as pd
import pytest

from training.build_leave_one_out_residual_continuation_2025_support import (
    END,
    HOLDOUT_START,
    assert_clock_contract,
    reserve_continuation_clock,
)


def candidates() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "signal_time": pd.Timestamp("2025-02-01 00:00"),
            "residual_horizon_hours": 12,
            "long_symbol": "ADAUSDT",
            "short_symbol": "ETHUSDT",
            "long_weight": 0.4,
            "short_weight_abs": 0.6,
            "long_beta": 1.5,
            "short_beta": 1.0,
            "loser_residual_z": -2.0,
            "winner_residual_z": 2.0,
            "loser_flow_z": 0.0,
            "winner_flow_z": 0.0,
            "exhaustion_score": 2.0,
            "eligible": True,
        },
        {
            "signal_time": pd.Timestamp("2025-02-01 06:00"),
            "residual_horizon_hours": 12,
            "long_symbol": "XRPUSDT",
            "short_symbol": "SOLUSDT",
            "long_weight": 0.5,
            "short_weight_abs": 0.5,
            "long_beta": 1.0,
            "short_beta": 1.0,
            "loser_residual_z": -2.1,
            "winner_residual_z": 2.1,
            "loser_flow_z": 0.0,
            "winner_flow_z": 0.0,
            "exhaustion_score": 2.1,
            "eligible": True,
        },
    ])


def test_reservation_flips_exact_lore_direction_and_preserves_beta_neutrality() -> None:
    index = pd.date_range("2025-01-01", "2025-02-02", freq="1h")
    quality = pd.Series(True, index=index)
    clock = reserve_continuation_clock(candidates(), quality)
    assert len(clock) == 1
    row = clock.iloc[0]
    assert row["long_symbol"] == "ETHUSDT"
    assert row["short_symbol"] == "ADAUSDT"
    assert row["long_weight"] == pytest.approx(0.6)
    assert row["short_weight_abs"] == pytest.approx(0.4)
    assert row["long_weight"] * row["long_beta"] == pytest.approx(
        row["short_weight_abs"] * row["short_beta"]
    )
    assert_clock_contract(clock)


def test_dirty_feature_window_removes_candidate() -> None:
    index = pd.date_range("2025-01-01", "2025-02-02", freq="1h")
    quality = pd.Series(True, index=index)
    quality.loc[pd.Timestamp("2025-01-31 20:00")] = False
    clock = reserve_continuation_clock(candidates().iloc[:1], quality)
    assert clock.empty


def test_clock_contract_rejects_future_exit() -> None:
    index = pd.date_range("2025-12-01", END, freq="1h")
    quality = pd.Series(True, index=index)
    frame = candidates().iloc[:1].copy()
    frame["signal_time"] = pd.Timestamp("2025-12-31 18:00")
    clock = reserve_continuation_clock(frame, quality)
    assert clock.empty


def test_boundaries_are_calendar_2025() -> None:
    assert HOLDOUT_START == pd.Timestamp("2025-01-01")
    assert END == pd.Timestamp("2026-01-01")
