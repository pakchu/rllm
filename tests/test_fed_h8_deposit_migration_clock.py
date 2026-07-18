from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from training import fed_h8_deposit_migration_clock as clock


def test_robust_z_is_strictly_prior_and_future_mutation_safe() -> None:
    source = pd.Series(np.linspace(-3.0, 4.0, 180))
    original = clock._robust_z(source, window=20)
    changed = source.copy()
    changed.iloc[150:] = changed.iloc[150:] * 1000
    replay = clock._robust_z(changed, window=20)
    pd.testing.assert_series_equal(original.iloc[:150], replay.iloc[:150])
    assert original.iloc[:20].isna().all()
    assert np.isfinite(original.iloc[20:]).all()


def test_tail_threshold_excludes_current_observation() -> None:
    score = pd.Series(np.arange(30, dtype=float))
    threshold = clock._tail_threshold(score, quantile=0.75, window=10)
    assert threshold.iloc[:10].isna().all()
    expected = np.quantile(np.arange(10, dtype=float), 0.75)
    assert threshold.iloc[10] == pytest.approx(expected)


def test_execution_is_1700_et_plus_exactly_48_hours() -> None:
    release = datetime(2023, 3, 17, 20, 15, tzinfo=timezone.utc)
    entry, exit_time = clock._execution_times(release)
    assert entry.isoformat() == "2023-03-17T21:00:00+00:00"
    assert exit_time.isoformat() == "2023-03-19T21:00:00+00:00"


def test_one_week_placebo_preserves_new_york_wall_clock_across_dst() -> None:
    release = datetime(2023, 3, 10, 21, 15, tzinfo=timezone.utc)
    entry, exit_time = clock._execution_times(release, delay_weeks=1)
    assert entry.isoformat() == "2023-03-17T21:00:00+00:00"
    assert exit_time.isoformat() == "2023-03-19T21:00:00+00:00"


def test_frozen_source_and_source_only_q50_counts() -> None:
    source = clock.load_source()
    events = clock.build_events(source, tail_quantile=0.50)
    stage1 = [event for event in events if "2020" <= event.entry_time[:4] <= "2022"]
    sealed_2023 = [event for event in events if event.entry_time.startswith("2023")]
    assert len(source) == 365
    assert len(events) == 99
    assert len(stage1) == 75
    assert sum(event.side == 1 for event in stage1) == 28
    assert sum(event.side == -1 for event in stage1) == 47
    assert len(sealed_2023) == 24
