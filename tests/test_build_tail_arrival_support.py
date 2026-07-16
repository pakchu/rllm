from __future__ import annotations

import numpy as np
import pandas as pd

from training import build_tail_arrival_support as support
from training.preregister_tail_arrival_absorption_alpha import Policy


def test_lagged_quantile_excludes_current_shock_and_ineligible_rows() -> None:
    values = pd.Series([1.0, 2.0, 100.0, 3.0, 4.0])
    eligible = pd.Series([True, True, True, False, True])
    result = support.lagged_quantile(
        values, eligible, 0.5, window=4, minimum=2
    )
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == 1.5
    assert result.iloc[3] == 2.0
    assert result.iloc[4] == 2.0


def test_missing_quarantine_is_current_plus_following_twenty_four() -> None:
    invalid = pd.Series([False] * 35)
    invalid.iloc[3] = True
    quarantined = support.missing_quarantine(invalid)
    assert quarantined.iloc[:3].eq(False).all()
    assert quarantined.iloc[3:28].eq(True).all()
    assert quarantined.iloc[28:].eq(False).all()


def test_episode_start_uses_prior_twelve_not_current() -> None:
    active = pd.Series([False] * 30)
    active.iloc[[2, 3, 14, 15, 28]] = True
    starts = support.episode_start(active)
    assert np.flatnonzero(starts).tolist() == [2, 28]


def test_branch_side_is_fade_for_absorption_and_follow_for_release() -> None:
    features = pd.DataFrame(
        {
            "packet_direction": [1, -1, 1],
            "absorption_start": [1, 1, 0],
            "release_start": [0, 1, 1],
        }
    )
    absorption = Policy("A", "tail_absorption_fade", 12)
    release = Policy("R", "tail_release_follow", 12)
    absorption_mask, absorption_side = support.branch_start_and_side(
        features, absorption
    )
    release_mask, release_side = support.branch_start_and_side(features, release)
    assert absorption_mask.tolist() == [True, True, False]
    assert absorption_side.tolist() == [-1, 1, -1]
    assert release_mask.tolist() == [False, True, True]
    assert release_side.tolist() == [1, -1, 1]


def test_nonoverlap_reserves_two_bar_entry_delay_and_exit_open() -> None:
    mask = np.ones(10, dtype=bool)
    assert support.schedule_nonoverlap(mask, 3).tolist() == [0, 3]
    assert support.schedule_nonoverlap(mask, 1).tolist() == list(range(7))


def test_support_fails_when_one_side_is_too_sparse() -> None:
    dates = pd.date_range("2020-01-01", periods=150, freq="7D")
    clocks = pd.DataFrame(
        {
            "policy_id": ["T"] * 150,
            "side": [1] * 140 + [-1] * 10,
            "signal_date": dates,
        }
    )
    metrics = support.support_metrics(
        clocks, Policy("T", "tail_absorption_fade", 12)
    )
    assert metrics["events_by_side"] == {"-1": 10, "1": 140}
    assert metrics["gates"]["each_side_share_min_0_20"] is False
    assert metrics["pass"] is False


def test_source_quality_uses_gap_and_following_quarantine() -> None:
    dates = pd.date_range("2020-01-01", periods=1000, freq="5min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "source_complete": np.ones(1000, dtype=np.int8),
            "source_gap_day": np.zeros(1000, dtype=np.int8),
            "source_quarantined": np.zeros(1000, dtype=np.int8),
        }
    )
    frame.loc[:4, "source_complete"] = 0
    frame.loc[:9, "source_gap_day"] = 1
    frame.loc[:59, "source_quarantined"] = 1
    metrics = support.source_quality(frame)
    assert metrics["source_missing_rows"] == 5
    assert metrics["source_gap_day_rows"] == 10
    assert metrics["missing_gap_or_following_24_quarantined_rows"] == 60
    assert metrics["global_missing_or_quarantined_fraction"] == 0.06
    assert metrics["pass"] is False
