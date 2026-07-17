from __future__ import annotations

import gzip
from pathlib import Path

from training import build_cboe_tail_hedge_disagreement_support as support


def test_control_clocks_are_source_only_and_short() -> None:
    clocks = support.build_control_events()
    assert tuple(clocks) == support.BASE_CLOCK_NAMES
    assert len(clocks["primary"]) >= 400
    assert all(event.side == "SHORT" for events in clocks.values() for event in events)


def test_support_passes_without_opening_outcomes(tmp_path: Path) -> None:
    output = tmp_path / "support.json"
    ledger = tmp_path / "clocks.csv.gz"
    report = support.build_support(output_path=output, ledger_path=ledger)
    assert report["support_passed"] is True
    assert report["advance_to_stage1_outcomes"] is True
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["market_rows_loaded"] == 0
    assert report["funding_rows_loaded"] == 0
    assert all(report["support_checks"].values())
    assert report["clocks"]["sha256"] == support.sha256_file(ledger)
    with gzip.open(ledger, "rt") as handle:
        assert handle.readline().strip().split(",") == list(support.LEDGER_COLUMNS)


def test_primary_clock_replays_preregistered_bytes() -> None:
    primary = support.build_control_events()["primary"]
    support._verify_primary_clock(primary)
