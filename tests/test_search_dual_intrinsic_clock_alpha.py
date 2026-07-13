from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_dual_intrinsic_clock_alpha import (
    WINDOWS,
    admission,
    build_paths,
    build_signals,
    directional_change_events,
    load_pre2024,
)


def test_directional_change_emits_at_most_one_event_per_bar() -> None:
    path = np.array([0.0, 2.0, -2.0, 2.0, -2.0])
    scale = np.ones(len(path))
    events = directional_change_events(path, scale, width=1.0)
    assert events.tolist() == [0, 1, -1, 1, -1]
    assert set(events).issubset({-1, 0, 1})


def test_directional_change_threshold_freezes_until_transition() -> None:
    path = np.array([0.0, 0.5, 0.9, 1.1])
    scale = np.array([1.0, 100.0, 100.0, 100.0])
    events = directional_change_events(path, scale, width=1.0)
    # The 1.0 threshold was frozen when state initialized; later scale spikes
    # cannot retroactively suppress the event.
    assert events.tolist() == [0, 0, 0, 1]


def test_directional_change_prefix_is_suffix_independent() -> None:
    path = np.sin(np.arange(100) / 3.0)
    scale = np.full(100, 0.25)
    left = directional_change_events(path, scale, width=1.0)
    right = directional_change_events(
        np.r_[path, [100.0, -100.0]], np.r_[scale, [0.01, 0.01]], width=1.0
    )[: len(path)]
    np.testing.assert_array_equal(left, right)


def test_build_paths_uses_prior_hour_quote_volume() -> None:
    size = 40
    frame = pd.DataFrame(
        {
            "close": 10_000.0 + np.arange(size),
            "quote_asset_volume": np.r_[np.ones(20), 1_000.0, np.ones(size - 21)],
            "taker_buy_quote": np.ones(size) * 0.75,
        }
    )
    paths = build_paths(frame)
    # At index 20 the denominator ends at index 19 and is exactly 12.
    expected = (2.0 * 0.75 - 1_000.0) / 12.0
    assert paths["flow_increment"][20] == expected


def test_impact_signal_maps_first_flow_fast_state_to_fade() -> None:
    size = 30
    features = pd.DataFrame(
        {
            "price_event_count": np.ones(size),
            "flow_event_count": np.r_[np.ones(23), np.full(7, 4.0)],
            "clock_log_ratio": np.zeros(size),
            "price_displacement_z": np.zeros(size),
            "flow_displacement_z": np.ones(size) * 2.0,
            "decision": np.r_[np.zeros(23, dtype=bool), True, np.zeros(6, dtype=bool)],
        }
    )
    long_signal, short_signal, diagnostics = build_signals(features, 2.0)
    assert not long_signal.any()
    assert np.flatnonzero(short_signal).tolist() == [23]
    assert diagnostics["state"][23] == 1


def test_persistent_state_does_not_reenter_with_first_entry_enabled() -> None:
    size = 37
    decision = np.zeros(size, dtype=bool)
    decision[[23, 35]] = True
    features = pd.DataFrame(
        {
            "price_event_count": np.ones(size),
            "flow_event_count": np.r_[np.ones(23), np.full(size - 23, 4.0)],
            "clock_log_ratio": np.zeros(size),
            "price_displacement_z": np.zeros(size),
            "flow_displacement_z": np.ones(size) * 2.0,
            "decision": decision,
        }
    )
    _, short_signal, _ = build_signals(features, 2.0)
    assert np.flatnonzero(short_signal).tolist() == [23]
    _, persistent_short, _ = build_signals(features, 2.0, first_entry_only=False)
    assert np.flatnonzero(persistent_short).tolist() == [23, 35]


def test_real_loader_physically_excludes_2024() -> None:
    _, dates = load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")
    assert dates.diff().dropna().eq(pd.Timedelta("5min")).all()


def test_admission_requires_fit_and_selection_ratio_three() -> None:
    def row(ratio: float, trades: int = 100) -> dict[str, float | int]:
        return {"return_pct": 1.0, "ratio": ratio, "trades": trades}

    stats = {name: row(3.1) for name in WINDOWS}
    assert admission(stats)
    stats["select_2023"] = row(2.99)
    assert not admission(stats)
