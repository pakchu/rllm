from __future__ import annotations

import pandas as pd

from training import build_btcdom_leverage_polarity_decomposition_support as support


def _events(entries: list[str], sides: list[int]) -> pd.DataFrame:
    dates = pd.to_datetime(entries, utc=True)
    return pd.DataFrame(
        {
            "entry_time": dates,
            "exit_time": dates + pd.Timedelta(hours=12),
            "side": sides,
        }
    )


def test_support_summary_reports_side_month_and_quarter_breadth() -> None:
    events = _events(
        ["2023-01-01 00:05", "2023-01-10 00:05", "2023-04-01 00:05"],
        [1, -1, 1],
    )
    summary = support.support_summary(events)
    assert summary["events"] == 3
    assert summary["long"] == 2
    assert summary["short"] == 1
    assert summary["month_counts"] == {"2023-01": 2, "2023-04": 1}
    assert summary["quarter_counts"] == {"2023Q1": 2, "2023Q2": 1}
    assert summary["max_month_share"] == 2 / 3


def test_novelty_metrics_are_bidirectional_and_year_bounded() -> None:
    primary = pd.DatetimeIndex(
        pd.to_datetime(
            ["2022-12-31 23:05", "2023-01-01 00:05", "2023-01-02 00:05"],
            utc=True,
        )
    )
    comparator = pd.DatetimeIndex(
        pd.to_datetime(
            ["2023-01-01 00:05", "2023-01-02 01:05", "2024-01-01 00:05"],
            utc=True,
        )
    )
    metrics = support.novelty_metrics(primary, comparator, year=2023, hours=1)
    assert metrics["primary_events"] == 2
    assert metrics["comparator_events"] == 2
    assert metrics["exact_intersection"] == 1
    assert metrics["exact_jaccard"] == 1 / 3
    assert metrics["primary_near_share"] == 1.0
    assert metrics["comparator_near_share"] == 1.0


def test_support_checks_require_every_year_quarter_side_and_novelty() -> None:
    prereg = support.load_preregistration()
    good = {
        "events": 120,
        "long_share": 0.5,
        "short_share": 0.5,
        "max_month_share": 0.15,
        "quarter_counts": {
            "2022Q1": 30,
            "2022Q2": 30,
            "2022Q3": 30,
            "2022Q4": 30,
        },
    }
    good_2023 = {**good, "quarter_counts": {f"2023Q{i}": 30 for i in range(1, 5)}}
    summaries = {"primary": {"2022": good, "2023": good_2023}}
    novelty = {
        "PSR-30/6": {"exact_jaccard": 0.01, "max_bidirectional_near_share": 0.10}
    }
    checks, failures = support.support_checks(summaries, novelty, prereg)
    assert failures == []
    assert all(checks.values())

    summaries["primary"]["2023"] = {**good_2023, "short_share": 0.20}
    novelty["PSR-30/6"] = {
        "exact_jaccard": 0.01,
        "max_bidirectional_near_share": 0.40,
    }
    _, failures = support.support_checks(summaries, novelty, prereg)
    assert "2023_short_share" in failures
    assert "PSR-30/6_near_containment" in failures


def test_clock_encoding_is_byte_reproducible() -> None:
    events = _events(["2023-01-01 00:05"], [1])
    assert support._clock_bytes(events) == support._clock_bytes(events)
