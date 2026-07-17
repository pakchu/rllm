from __future__ import annotations

import gzip
import json
from pathlib import Path

from training import build_cboe_institutional_hedge_migration_support as support


def test_control_clocks_are_source_only_and_short() -> None:
    clocks = support.build_control_events()
    assert tuple(clocks) == support.BASE_CLOCK_NAMES
    assert len(clocks["primary"]) == 258
    assert all(event.side == "SHORT" for events in clocks.values() for event in events)


def test_support_passes_without_opening_outcomes(tmp_path: Path) -> None:
    output = tmp_path / "support.json"
    ledger = tmp_path / "clocks.csv.gz"
    report = support.build_support(output_path=output, ledger_path=ledger)
    support.validate_support(report)
    assert report["support_passed"] is True
    assert report["advance_to_stage1_outcomes"] is True
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["market_rows_loaded"] == 0
    assert report["funding_rows_loaded"] == 0
    assert all(report["checks"].values())
    assert report["clocks"]["sha256"] == support.sha256_file(ledger)
    with gzip.open(ledger, "rt") as handle:
        assert handle.readline().strip().split(",") == list(support.LEDGER_COLUMNS)


def test_primary_clock_replays_preregistered_bytes() -> None:
    primary = support.build_control_events()["primary"]
    support._verify_primary_clock(primary)


def test_frozen_support_artifacts_are_bound() -> None:
    result_path = Path(support.DEFAULT_OUTPUT)
    ledger_path = Path(support.DEFAULT_LEDGER)
    assert support.sha256_file(result_path) == (
        "df7ce11f92d50c52ffd61afd7cab6a2a3da5696e08ba45b886e3c37bec7f0a93"
    )
    assert support.sha256_file(ledger_path) == (
        "5e04cffacb1754c3111fcc32b09d72f06b546a4803b40c77d655a9787b015c0b"
    )
    payload = json.loads(result_path.read_text())
    support.validate_support(payload)
    assert payload["manifest_hash"] == (
        "2fb3872b8e062e4719d3b3f087857561262c7c2bdbccb06eaf780923bb051ae1"
    )
