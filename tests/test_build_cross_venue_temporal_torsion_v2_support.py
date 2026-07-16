from __future__ import annotations

import numpy as np
import pandas as pd

from training import build_cross_venue_temporal_torsion_v2_support as support
from training.preregister_cross_venue_temporal_torsion_alpha_v2 import Policy


def test_lagged_route_quantile_excludes_current_and_separates_counts() -> None:
    score = pd.Series([1.0, 2.0, 100.0, 3.0, 4.0])
    eligible = pd.Series([True, True, True, False, True])
    clean = pd.Series([True] * 5)
    threshold, clean_count, event_count = support.lagged_route_quantile(
        score,
        eligible,
        clean,
        window=4,
        minimum_clean=2,
        minimum_events=2,
        quantile=0.5,
    )
    assert threshold.iloc[:2].isna().all()
    assert threshold.iloc[2] == 1.5
    assert threshold.iloc[3] == 2.0
    assert threshold.iloc[4] == 2.0
    assert clean_count.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert event_count.tolist() == [0.0, 1.0, 2.0, 3.0, 3.0]


def test_episode_start_uses_prior_twelve_completed_bars() -> None:
    active = pd.Series([False] * 30)
    active.iloc[[2, 3, 14, 15, 28]] = True
    assert np.flatnonzero(support.episode_start(active)).tolist() == [2, 28]


def test_route_side_uses_source_venue() -> None:
    features = pd.DataFrame(
        {
            "spot_to_um_start": [1, 0],
            "um_to_spot_start": [0, 1],
            "spot_source_side": [1, -1],
            "um_source_side": [-1, -1],
        }
    )
    spot = Policy("S", "spot_preload_um_echo", 6)
    um = Policy("U", "um_preload_spot_echo", 6)
    spot_mask, spot_side = support.route_start_and_side(features, spot)
    um_mask, um_side = support.route_start_and_side(features, um)
    assert spot_mask.tolist() == [True, False]
    assert spot_side.tolist() == [1, -1]
    assert um_mask.tolist() == [False, True]
    assert um_side.tolist() == [-1, -1]


def test_nonoverlap_reserves_ten_minute_entry_and_exit_open() -> None:
    mask = np.ones(12, dtype=bool)
    assert support.schedule_nonoverlap(mask, 3).tolist() == [0, 3, 6]
    assert support.schedule_nonoverlap(mask, 1).tolist() == list(range(9))


def test_support_fails_side_imbalance() -> None:
    dates = pd.date_range("2020-01-01", periods=700, freq="36h")
    clocks = pd.DataFrame(
        {
            "policy_id": ["V"] * 700,
            "signal_date": dates,
            "side": [1] * 650 + [-1] * 50,
        }
    )
    metrics = support.support_metrics(
        clocks, Policy("V", "spot_preload_um_echo", 6)
    )
    assert metrics["gates"]["each_side_share_min_0_35"] is False
    assert metrics["pass"] is False


def test_source_quality_accepts_preregistered_v2_bounds() -> None:
    dates = pd.date_range("2020-01-01", periods=10_000, freq="5min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "source_valid_current": np.ones(10_000, dtype=np.int8),
            "source_quarantined": np.zeros(10_000, dtype=np.int8),
        }
    )
    frame.loc[:39, "source_valid_current"] = 0
    frame.loc[:39, "source_quarantined"] = 1
    metrics = support.source_quality(frame)
    assert metrics["global_fraction"] == 0.004
    assert metrics["pass"] is True
