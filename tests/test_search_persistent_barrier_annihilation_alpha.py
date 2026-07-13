from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_persistent_barrier_annihilation_alpha import (
    WINDOWS,
    admission,
    build_signals,
    fit_threshold,
    frozen_rolling_extrema_signals,
    load_pre2024,
    persistent_barrier_features,
)


def hourly_dates(size: int) -> pd.Series:
    return pd.Series(pd.date_range("2020-01-01", periods=size, freq="5min"))


def test_upward_persistent_peak_mass_is_observed_after_frozen_traversal() -> None:
    # The frozen history before index 24 has two local peaks.  The traversal
    # from 24 to 36 crosses the lower peak but not the higher one.
    price = np.array(
        [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 4, 0, 4,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5] + [5] * 11,
        dtype=float,
    )
    # Patch the long prior-scale warm-up with a longer deterministic prefix.
    prefix = np.sin(np.arange(1104) / 7.0) * 0.1
    values = np.r_[prefix, price]
    dates = hourly_dates(len(values))
    features = persistent_barrier_features(
        values,
        dates,
        horizon=24,
        minimum_prominence_z=0.0,
    )
    decision = 1104 + 36
    assert features.loc[decision, "side"] == 1
    assert features.loc[decision, "barrier_count"] >= 1
    assert features.loc[decision, "persistence_mass"] > 0.0


def test_downward_traversal_uses_persistent_troughs() -> None:
    prefix = np.sin(np.arange(1104) / 7.0) * 0.1
    tail = np.array([0, -1, 0, -2, 0, -3, 0, -4, 0, -5, 0, -6, 0, -7, 0, -8, 0, -9, 0, -10, 0, -4, 0, -4] + [-3] * 12 + [-5] + [-5] * 11)
    values = np.r_[prefix, tail]
    dates = hourly_dates(len(values))
    features = persistent_barrier_features(values, dates, horizon=24, minimum_prominence_z=0.0)
    decision = 1104 + 36
    assert features.loc[decision, "side"] == -1
    assert features.loc[decision, "barrier_count"] >= 1


def test_feature_prefix_is_suffix_independent() -> None:
    size = 1300
    values = 0.001 * np.arange(size) + np.sin(np.arange(size) / 9.0)
    dates = hourly_dates(size)
    left = persistent_barrier_features(values, dates, horizon=24, minimum_prominence_z=0.0)
    extended_values = np.r_[values, np.linspace(100.0, 200.0, 24)]
    extended_dates = hourly_dates(len(extended_values))
    right = persistent_barrier_features(
        extended_values, extended_dates, horizon=24, minimum_prominence_z=0.0
    ).iloc[:size]
    pd.testing.assert_frame_equal(left, right.reset_index(drop=True))


def test_fit_threshold_never_reads_outside_mask() -> None:
    values = np.r_[np.arange(1.0, 102.0), 1_000_000.0]
    mask = np.r_[np.ones(101, dtype=bool), False]
    threshold, count = fit_threshold(values, mask, 0.5)
    assert threshold == 51.0
    assert count == 101


def test_signal_direction_modes_are_exact_flips() -> None:
    features = pd.DataFrame(
        {
            "side": [0, 1, -1],
            "persistence_mass": [np.nan, 2.0, 3.0],
        }
    )
    long_signal, short_signal = build_signals(features, "persistence_mass", 1.0, "continue")
    fade_long, fade_short = build_signals(features, "persistence_mass", 1.0, "fade")
    assert np.array_equal(long_signal, fade_short)
    assert np.array_equal(short_signal, fade_long)


def test_frozen_global_extrema_control_excludes_current_path() -> None:
    values = np.r_[np.zeros(24), 1.0, np.ones(11), 2.0, np.ones(11)]
    dates = hourly_dates(len(values))
    long_active, short_active = frozen_rolling_extrema_signals(values, dates, horizon=24)
    # At index 36 the prior frozen maximum was zero and the start was already
    # above it, so the current move cannot retroactively count it as a cross.
    assert not long_active[36]
    assert not short_active[36]


def test_real_loader_physically_excludes_2024() -> None:
    _, dates = load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")
    assert dates.diff().dropna().eq(pd.Timedelta("5min")).all()


def test_admission_requires_fit_and_selection_ratio_three() -> None:
    def row(ratio: float, trades: int = 100) -> dict[str, float | int]:
        return {"return_pct": 1.0, "ratio": ratio, "trades": trades}

    stats = {name: row(3.1) for name in WINDOWS}
    assert admission(stats)
    stats["fit"] = row(2.99)
    assert not admission(stats)
