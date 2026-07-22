from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import build_federal_liquidity_narrative_sponsorship_support as s


def test_midrank_numerator_excludes_current_and_counts_ties() -> None:
    prior = list(range(103)) + [200]
    assert s.midrank_numerator(50, prior) == 2 * 50 + 1


def test_narrative_quality_is_finite_on_audited_zero_row() -> None:
    frame = pd.DataFrame(
        {
            "adoption_article_count": [0, 10],
            "failure_article_count": [0, 3],
            "constraint_article_count": [0, 2],
        }
    )
    quality = s.narrative_quality(frame)
    assert quality.notna().all()
    assert quality.iloc[0] == pytest.approx(np.log(0.5))


def test_schedule_allows_same_time_roll_but_suppresses_overlap() -> None:
    frame = pd.DataFrame(
        {
            "release_date": pd.to_datetime(["2022-01-01", "2022-01-08", "2022-01-14"], utc=True),
            "narrative_source_date": pd.to_datetime(["2021-12-30", "2022-01-06", "2022-01-12"], utc=True),
            "signal_time": pd.to_datetime(["2022-01-01", "2022-01-08", "2022-01-14"], utc=True),
            "entry_time": pd.to_datetime(["2022-01-01", "2022-01-08", "2022-01-14"], utc=True),
            "exit_time": pd.to_datetime(["2022-01-08", "2022-01-15", "2022-01-21"], utc=True),
            "test_side": ["LONG", "SHORT", "LONG"],
        }
    )
    result = s._schedule(frame, "test", "test_side")
    assert len(result) == 2
    assert result["entry_time"].tolist() == [
        pd.Timestamp("2022-01-01T00:00:00Z"),
        pd.Timestamp("2022-01-08T00:00:00Z"),
    ]


def test_one_to_one_overlap_uses_each_comparator_once_and_normalizes_side() -> None:
    primary = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2022-01-01T00:10Z", "2022-01-01T00:20Z"]),
            "side": ["LONG", "SHORT"],
        }
    )
    comparator = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2022-01-01T00:15Z"]),
            "side": [1],
        }
    )
    stats = s.one_to_one_overlap(primary, comparator)
    assert stats["matched"] == 1
    assert stats["same_side_matched"] == 1
    assert stats["flnsr_containment"] == 0.5


def test_clock_schema_contains_no_market_or_outcome_fields() -> None:
    assert not {
        "open",
        "high",
        "low",
        "close",
        "return",
        "pnl",
        "funding",
    }.intersection(s.CLOCK_COLUMNS)
