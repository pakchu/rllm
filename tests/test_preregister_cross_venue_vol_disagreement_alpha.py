from __future__ import annotations

import numpy as np
import pandas as pd

from training import preregister_cross_venue_vol_disagreement_alpha as prereg


def _frame(periods: int = 220) -> pd.DataFrame:
    signal = pd.date_range("2023-06-20", periods=periods, freq="1h")
    return pd.DataFrame(
        {
            "signal_time_utc": signal,
            "feature_available_time_utc": signal,
            "trade_earliest_time_utc": signal + pd.Timedelta("5min"),
            "log_bvol_dvol_ratio": np.linspace(-1.0, 1.0, periods),
            "btc_return_4h": np.tile([-0.04, 0.04], periods // 2),
            "feature_valid": True,
        }
    )


def test_candidate_grid_is_frozen_at_24_cells() -> None:
    assert len(prereg.CANDIDATES) == 24
    assert len({candidate.name for candidate in prereg.CANDIDATES}) == 24


def test_prior_quantile_excludes_current_observation() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 1000.0])
    valid = pd.Series(True, index=values.index)
    threshold = prereg.prior_quantile(values, valid, 0.5, window=4, min_periods=3)
    assert threshold.iloc[3] == 2.0


def test_family_directions_are_fade_and_follow() -> None:
    frame = _frame()
    thresholds = prereg.build_thresholds(frame)
    fade = prereg.Candidate("bvol_rich_move_fade", 0.8, 0.8, 12)
    follow = prereg.Candidate("dvol_rich_move_follow", 0.8, 0.8, 12)
    fade_onset, fade_side = prereg.candidate_clock(frame, thresholds, fade)
    follow_onset, follow_side = prereg.candidate_clock(frame, thresholds, follow)

    for position in np.flatnonzero(fade_onset.to_numpy()):
        assert fade_side.iloc[position] == -np.sign(frame["btc_return_4h"].iloc[position])
    for position in np.flatnonzero(follow_onset.to_numpy()):
        assert follow_side.iloc[position] == np.sign(frame["btc_return_4h"].iloc[position])


def test_schedule_delays_entry_and_enforces_nonoverlap() -> None:
    frame = _frame(10)
    onset = pd.Series([True, True, False, True, False, False, False, False, False, False])
    side = pd.Series(1, index=frame.index, dtype="int8")
    schedule = prereg.nonoverlapping_schedule(frame, onset, side, hold_hours=2)

    assert len(schedule) == 2
    assert pd.Timestamp(schedule.iloc[0]["entry_time"]) == frame.iloc[0]["signal_time_utc"] + pd.Timedelta("5min")
    assert pd.Timestamp(schedule.iloc[1]["entry_time"]) >= pd.Timestamp(schedule.iloc[0]["exit_time"])
