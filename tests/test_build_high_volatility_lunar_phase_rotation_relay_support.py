import numpy as np
import pandas as pd

from training import build_high_volatility_lunar_phase_rotation_relay_support as b


def test_strict_prior_midrank_excludes_current():
    values = pd.Series([1.0, 2.0, 3.0])
    ranks = b.strict_prior_midrank(values, lookback=2, minimum=2)
    assert ranks.iloc[:2].isna().all()
    assert ranks.iloc[2] == 1.0


def test_phase_window_maps_published_direction_and_excludes_quarters():
    phases = pd.DataFrame({
        "phase": ["New Moon", "Full Moon"],
        "phase_time": [pd.Timestamp("2023-07-01T12:00:00Z"), pd.Timestamp("2023-07-16T12:00:00Z")],
    })
    daily = pd.DataFrame({
        "source_day": pd.date_range("2023-06-29", periods=20, freq="1D", tz="UTC"),
        "realized_variation": np.arange(1.0, 21.0),
    })
    features = b.build_features(phases, daily)
    assert features.loc[features.decision_time.eq(pd.Timestamp("2023-07-01T00:00:00Z")), "phase_side"].item() == 1
    assert features.loc[features.decision_time.eq(pd.Timestamp("2023-07-16T00:00:00Z")), "phase_side"].item() == -1
    assert features.loc[features.decision_time.eq(pd.Timestamp("2023-07-09T00:00:00Z")), "phase_side"].item() == 0


def test_primary_clock_uses_five_minute_entry_and_twenty_four_hour_hold():
    frame = pd.DataFrame({
        "decision_time": [pd.Timestamp("2023-07-01T00:00:00Z")],
        "phase": ["New Moon"], "phase_time": [pd.Timestamp("2023-07-01T12:00:00Z")],
        "phase_distance_hours": [12.0], "phase_side": [1],
        "btc_realized_variation": [0.03], "btc_variation_rank": [0.9],
    })
    clock = b.build_clock(frame)
    assert clock.entry_time.iloc[0] == pd.Timestamp("2023-07-01T00:05:00Z")
    assert clock.exit_time.iloc[0] == pd.Timestamp("2023-07-02T00:05:00Z")
    assert clock.side.iloc[0] == 1


def test_direction_flip_is_diagnostic_only():
    frame = pd.DataFrame({
        "decision_time": [pd.Timestamp("2023-07-01T00:00:00Z")],
        "phase": ["Full Moon"], "phase_time": [pd.Timestamp("2023-07-01T12:00:00Z")],
        "phase_distance_hours": [12.0], "phase_side": [-1],
        "btc_realized_variation": [0.03], "btc_variation_rank": [0.9],
    })
    assert b.build_clock(frame).side.iloc[0] == -1
    assert b.build_clock(frame, "phase_direction_flip").side.iloc[0] == 1
