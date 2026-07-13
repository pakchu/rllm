from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_oi_roundtrip_geometric_phase_alpha import (
    WINDOWS,
    admission,
    build_bar_state,
    build_signals,
    fit_threshold,
    load_pre2024,
    oi_roundtrip_features,
)


def synthetic_state(size: int = 32, recross: int = 12) -> dict[str, np.ndarray]:
    log_price = np.linspace(0.0, 0.031, size)
    log_oi = np.zeros(size)
    log_oi[1:recross] = 0.10
    oi_delta = np.diff(log_oi, prepend=np.nan)
    oi_delta_z = np.zeros(size)
    oi_delta_z[1] = 3.0
    return {
        "log_price": log_price,
        "price_return": np.diff(log_price, prepend=np.nan),
        "prior_price_vol": np.full(size, 0.01),
        "log_oi": log_oi,
        "oi_delta": oi_delta,
        "prior_oi_scale": np.full(size, 0.01),
        "oi_delta_z": oi_delta_z,
        "available": np.ones(size, dtype=bool),
    }


def test_first_oi_reclosure_emits_once_at_completed_bar() -> None:
    frame = oi_roundtrip_features(synthetic_state(), departure_z=2.0, max_age=24)
    emitted = np.flatnonzero(frame["episode_age"].to_numpy(int) > 0)
    assert emitted.tolist() == [12]
    assert frame.loc[12, "episode_age"] == 12
    assert frame.loc[12, "residual_side"] == 1
    assert frame.loc[12, "residual_phase"] > 0.0
    assert frame.loc[12, "terminal_persistence"] > 0.0


def test_reclosure_before_minimum_age_is_discarded() -> None:
    frame = oi_roundtrip_features(
        synthetic_state(recross=6),
        departure_z=2.0,
        max_age=24,
        min_age=12,
    )
    assert int((frame["episode_age"] > 0).sum()) == 0


def test_fixed_age_control_ignores_early_reclosure_causally() -> None:
    frame = oi_roundtrip_features(
        synthetic_state(size=40, recross=6),
        departure_z=2.0,
        max_age=24,
        ignore_first_return=True,
    )
    emitted = np.flatnonzero(frame["episode_age"].to_numpy(int) > 0)
    assert emitted.tolist() == [24]


def test_feature_prefix_is_suffix_independent() -> None:
    prefix = synthetic_state(size=32, recross=12)
    extended = {name: np.r_[values, values[-8:]] for name, values in prefix.items()}
    left = oi_roundtrip_features(prefix, departure_z=2.0, max_age=24)
    right = oi_roundtrip_features(extended, departure_z=2.0, max_age=24).iloc[: len(left)]
    pd.testing.assert_frame_equal(left, right.reset_index(drop=True))


def test_signal_mapping_and_exact_flip() -> None:
    features = pd.DataFrame(
        {
            "residual_phase": [np.nan, 2.0, 3.0],
            "terminal_persistence": [np.nan, 2.0, 3.0],
            "inventory_work": [np.nan, 4.0, 5.0],
            "residual_side": [0, 1, -1],
            "work_side": [0, -1, 1],
        }
    )
    long_signal, short_signal = build_signals(features, "residual_phase", 1.0)
    assert long_signal.tolist() == [False, True, False]
    assert short_signal.tolist() == [False, False, True]
    flip_long, flip_short = build_signals(features, "residual_phase", 1.0, flip=True)
    assert np.array_equal(flip_long, short_signal)
    assert np.array_equal(flip_short, long_signal)
    work_long, work_short = build_signals(features, "inventory_work", 1.0)
    assert work_long.tolist() == [False, False, True]
    assert work_short.tolist() == [False, True, False]


def test_fit_threshold_never_reads_outside_mask() -> None:
    values = np.r_[np.arange(1.0, 22.0), 1_000_000.0]
    fit_mask = np.r_[np.ones(21, dtype=bool), False]
    threshold, count = fit_threshold(values, fit_mask, 0.5)
    assert threshold == 11.0
    assert count == 21


def test_build_bar_state_is_prefix_causal() -> None:
    size = 2600
    index = np.arange(size, dtype=float)
    base = pd.DataFrame(
        {
            "close": 20_000.0 * np.exp(0.00005 * index + 0.001 * np.sin(index / 17.0)),
            "open_interest": 100_000.0 * np.exp(0.00002 * index + 0.001 * np.cos(index / 13.0)),
            "open_interest_available": np.ones(size),
        }
    )
    extension = pd.DataFrame(
        {
            "close": [1.0] * 20,
            "open_interest": [1.0] * 20,
            "open_interest_available": [1.0] * 20,
        }
    )
    left = build_bar_state(base)
    right = build_bar_state(pd.concat([base, extension], ignore_index=True))
    for name in left:
        np.testing.assert_allclose(left[name], right[name][:size], equal_nan=True)


def test_real_loader_physically_excludes_2024() -> None:
    _, dates = load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")
    assert dates.diff().dropna().eq(pd.Timedelta("5min")).all()


def test_admission_requires_both_period_ratios() -> None:
    def row(ratio: float, trades: int = 50) -> dict[str, float | int]:
        return {
            "return_pct": 1.0,
            "ratio": ratio,
            "trades": trades,
            "longs": trades // 2,
            "shorts": trades - trades // 2,
        }

    stats = {name: row(3.1) for name in WINDOWS}
    assert admission(stats)
    stats["select_2023"] = row(2.99)
    assert not admission(stats)
