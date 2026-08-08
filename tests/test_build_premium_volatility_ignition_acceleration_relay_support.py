import numpy as np
import pandas as pd

from training import build_premium_volatility_ignition_acceleration_relay_support as support


def raw_hour(start: str, *, missing: int | None = None) -> pd.DataFrame:
    ts = pd.date_range(start, periods=60, freq="1min")
    frame = pd.DataFrame({
        "ts": ts,
        "open": np.arange(60, dtype=float),
        "high": np.arange(60, dtype=float) + 2.0,
        "low": np.arange(60, dtype=float) - 2.0,
        "close": np.arange(60, dtype=float) + 1.0,
        "close_time": np.arange(60),
        "row_valid": True,
    })
    return frame.drop(index=missing).reset_index(drop=True) if missing is not None else frame


def signal_frame() -> pd.DataFrame:
    decision = pd.Timestamp("2024-07-01T08:00:00Z")
    return pd.DataFrame({
        "decision_time": [decision - pd.Timedelta(hours=1), decision],
        "signal_valid": [True, True],
        "bvol_body": [-0.01, 0.02],
        "dvol_body": [-0.01, 0.03],
        "first_half_move": [0.1, 2.0],
        "second_half_move": [-0.1, 3.0],
        "prior_abs_first_half_q60": [1.0, 1.0],
    })


def test_hourly_premium_features_requires_all_exact_minute_offsets(monkeypatch):
    monkeypatch.setattr(support, "START", pd.Timestamp("2024-01-01T00:00:00Z"))
    monkeypatch.setattr(support, "END", pd.Timestamp("2024-01-01T01:00:00Z"))
    complete = support.hourly_premium_features(raw_hour("2024-01-01T00:00:00Z"))
    assert bool(complete.iloc[0].premium_valid)
    assert complete.iloc[0].first_half_move == 30.0
    assert complete.iloc[0].second_half_move == 30.0
    incomplete = support.hourly_premium_features(raw_hour("2024-01-01T00:00:00Z", missing=29))
    assert not bool(incomplete.iloc[0].premium_valid)
    assert pd.isna(incomplete.iloc[0].first_half_move)


def test_pviar_follows_same_direction_premium_acceleration():
    clock = support.build_clock(signal_frame())
    assert len(clock) == 1 and clock.iloc[0].side == 1
    assert clock.iloc[0].entry_time == pd.Timestamp("2024-07-01T08:05:00Z")
    assert clock.iloc[0].exit_time - clock.iloc[0].entry_time == pd.Timedelta(hours=6)


def test_pviar_rejects_absorption_deceleration_and_single_venue_expansion():
    frame = signal_frame()
    frame.loc[1, "second_half_move"] = -3.0
    assert support.build_clock(frame).empty
    frame = signal_frame()
    frame.loc[1, "second_half_move"] = 1.0
    assert support.build_clock(frame).empty
    frame = signal_frame()
    frame.loc[1, "dvol_body"] = -0.01
    assert support.build_clock(frame).empty


def test_pviar_direction_flip_is_clock_identical():
    primary = support.build_clock(signal_frame())
    flipped = support.build_clock(signal_frame(), "direction_flip")
    assert len(primary) == len(flipped) == 1
    assert primary.iloc[0].side == -flipped.iloc[0].side
    assert primary.iloc[0].entry_time == flipped.iloc[0].entry_time
