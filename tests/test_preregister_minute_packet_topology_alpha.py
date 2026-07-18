from __future__ import annotations

import numpy as np
import pandas as pd

from training import preregister_minute_packet_topology_alpha as alpha


def test_candidate_grid_is_small_and_frozen() -> None:
    assert len(alpha.CANDIDATES) == 24
    assert {candidate.family for candidate in alpha.CANDIDATES} == {
        "um_swarm_absorption",
        "cross_venue_churn_breakout",
    }
    assert {candidate.hold_bars for candidate in alpha.CANDIDATES} == {24, 48, 96}


def test_rolling_threshold_excludes_current_and_future_rows() -> None:
    values = pd.Series(np.arange(10, dtype=float))
    valid = pd.Series(True, index=values.index)
    baseline = alpha.rolling_prior_quantile(
        values, valid, 0.5, window=4, min_periods=2
    )
    changed = values.copy()
    changed.iloc[5:] = 1_000_000.0
    replay = alpha.rolling_prior_quantile(
        changed, valid, 0.5, window=4, min_periods=2
    )

    assert baseline.iloc[5] == replay.iloc[5]
    assert baseline.iloc[5] == 2.5
    assert pd.isna(baseline.iloc[1])


def test_nonoverlapping_schedule_uses_next_open_and_fixed_exit() -> None:
    frame = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=20, freq="5min")})
    onset = pd.Series(False, index=frame.index)
    onset.iloc[[2, 3, 8]] = True
    side = pd.Series(0, index=frame.index, dtype=np.int8)
    side.iloc[[2, 3, 8]] = np.asarray([1, -1, -1], dtype=np.int8)
    schedule = alpha.nonoverlapping_schedule(frame, onset, side, hold_bars=4)

    assert schedule[["signal_position", "entry_position", "exit_position", "side"]].to_dict(
        orient="records"
    ) == [
        {"signal_position": 2, "entry_position": 3, "exit_position": 7, "side": 1},
        {"signal_position": 8, "entry_position": 9, "exit_position": 13, "side": -1},
    ]


def test_support_gates_enforce_era_side_and_concentration_floors() -> None:
    summary = {
        "total": 200,
        "longs": 90,
        "shorts": 110,
        "by_year": {"2020": 50, "2021": 50, "2022": 50, "2023": 50},
        "by_2023_half": {"h1": 25, "h2": 25},
        "max_month_fraction": 0.08,
    }
    assert all(alpha.support_gates(summary).values())
    summary["shorts"] = 20
    assert not alpha.support_gates(summary)["each_side_at_least_25pct"]
