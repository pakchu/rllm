import numpy as np
import pandas as pd

from training import audit_legacy_dollar_rally_short as audit


def test_legacy_top0_is_frozen_expected_candidate():
    row = audit.legacy_top0()
    assert row["hold"] == 144
    assert row["stride"] == 12
    assert {g["feature"] for g in row["gates"]} == {"dxy_momentum", "htf_1d_return_4"}


def test_global_signal_positions_use_legacy_phase_and_next_bar_entry():
    positions = audit._global_signal_positions(400, hold_bars=144, stride_bars=12)
    assert positions[:3].tolist() == [143, 155, 167]
    assert np.all(np.diff(positions) == 12)
    assert (positions[:3] + audit.ENTRY_DELAY_BARS).tolist() == [144, 156, 168]


def test_window_entries_original_and_availability_variant():
    dates = pd.date_range("2024-01-01", periods=400, freq="5min")
    features = pd.DataFrame(
        {
            "dxy_momentum": np.full(400, 0.003),
            "htf_1d_return_4": np.full(400, 0.02),
            "dxy_available": np.zeros(400),
        }
    )
    entries, availability = audit._window_entries(
        pd.DatetimeIndex(dates), features, start="2024-01-01", end="2024-01-02", variant="original"
    )
    assert entries.tolist() == [144, 156, 168, 180, 192, 204, 216, 228, 240, 252]
    assert availability["dxy_unavailable_at_signal_count"] == len(entries)
    available_entries, available_diag = audit._window_entries(
        pd.DatetimeIndex(dates),
        features,
        start="2024-01-01",
        end="2024-01-02",
        variant="require_dxy_available_at_signal",
    )
    assert available_entries.tolist() == []
    assert available_diag["candidate_signal_count_before_schedule"] == 0
