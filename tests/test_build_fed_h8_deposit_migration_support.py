from __future__ import annotations

import json

from training import build_fed_h8_deposit_migration_support as support
from training import fed_h8_deposit_migration_clock as clock


def test_control_ledger_contains_all_frozen_controls() -> None:
    rows = support.build_clock_rows(clock.load_source())
    grouped = {
        name: [row for row in rows if row["control"] == name]
        for name in support.CONTROL_NAMES
    }
    assert all(grouped.values())
    assert len(grouped["primary"]) == 99
    assert len(grouped["direction_flip"]) == 99
    assert all(
        flipped["side"] == -original["side"]
        for original, flipped in zip(grouped["primary"], grouped["direction_flip"])
    )
    assert not set(clock.STRUCTURAL_EXCLUSION_RELEASES) & {
        row["release_date"] for row in rows
    }


def test_support_build_opens_no_outcomes(tmp_path) -> None:
    report = support.build_report(
        clocks_path=tmp_path / "controls.csv.gz", write_clock=True
    )
    support.validate_report(report)
    assert report["support_passed"] is True
    assert report["advance_to_stage1_outcomes"] is True
    assert report["market_or_funding_rows_opened"] == 0
    assert report["outcome_sources_opened"] == []
    primary = report["clock_summaries"]["primary"]
    assert primary["stage1"]["events"] == 75
    assert primary["2023"]["events"] == 24
    assert all(report["support_checks"].values())


def test_frozen_support_report_is_self_sealed() -> None:
    report = json.loads(open(support.DEFAULT_OUTPUT).read())
    support.validate_report(report)
    assert report["preregistration_sha256"] == support.PREREGISTRATION_SHA256
    assert report["clocks"]["sha256"] == support._sha256(support.DEFAULT_CLOCKS)
