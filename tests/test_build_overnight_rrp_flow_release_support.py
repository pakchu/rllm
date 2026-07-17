from __future__ import annotations

import hashlib

from training import build_overnight_rrp_flow_release_support as support


def test_support_is_outcome_blind_and_passes() -> None:
    report = support.build_report()
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["market_or_funding_rows_opened"] == 0
    assert report["support_passed"] is True
    assert all(report["support_checks"].values())
    assert report["source_rows"] == 1498
    assert report["source_quarantined_rows"] == 9


def test_primary_and_control_counts_are_frozen() -> None:
    report = support.build_report(write_clock=False)
    primary = report["clock_summaries"]["primary"]
    delta = report["clock_summaries"]["one_day_delta_tail"]
    delayed = report["clock_summaries"]["one_release_delay"]
    assert primary["train"] == {
        "events": 112,
        "longs": 63,
        "shorts": 49,
        "max_single_month_count": 16,
        "max_single_month_share": 1 / 7,
    }
    assert primary["2023"]["events"] == 74
    assert delta["train"]["events"] == 99
    assert delayed["train"]["events"] == 112


def test_clock_artifact_hash_matches_report() -> None:
    report = support.build_report()
    expected = hashlib.sha256(
        support.Path(support.DEFAULT_CLOCKS).read_bytes()
    ).hexdigest()
    assert report["clocks"]["sha256"] == expected


def test_delayed_control_is_exactly_one_operation_later() -> None:
    source_rows = support.clock.read_source()
    ledger = support.build_clock_rows(source_rows)
    primary = [row for row in ledger if row["control"] == "primary"]
    delayed = [row for row in ledger if row["control"] == "one_release_delay"]
    assert len(delayed) in {len(primary), len(primary) - 1}
    assert delayed[0]["entry_time"] == primary[0]["exit_time"]
    assert delayed[0]["side"] == primary[0]["side"]
