from __future__ import annotations

import json
from pathlib import Path

from training import preregister_inflation_breadth_release_drift as prereg


def test_preregistration_opens_no_outcome(tmp_path: Path) -> None:
    report = prereg.run(
        tmp_path / "registration.json",
        tmp_path / "clock.csv.gz",
        tmp_path / "registration.md",
    )
    assert report["policy_id"] == "IBRD-7"
    assert report["source_contract"]["market_or_funding_rows_loaded"] == 0
    assert report["opened_outcome_windows"] == []
    assert report["simulation_run"] is False
    assert report["policy"]["mutable_parameters"] == []
    assert report["source_only_distributions"]["stage1_2020_2022"] == {
        "trades": 20,
        "longs": 8,
        "shorts": 12,
        "max_single_month_count": 1,
        "max_single_month_share": 0.05,
    }


def test_frozen_preregistration_artifact_replays() -> None:
    stored = json.loads(Path(prereg.DEFAULT_OUTPUT).read_text())
    replay = prereg.build_report(Path(prereg.DEFAULT_CLOCK))
    assert replay == stored
