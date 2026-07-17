from __future__ import annotations

import json
from pathlib import Path

from training import preregister_eia_petroleum_stock_breadth as prereg


def test_preregistration_opens_no_outcome(tmp_path: Path) -> None:
    report = prereg.run(
        tmp_path / "registration.json",
        tmp_path / "clock.csv.gz",
        tmp_path / "registration.md",
    )
    assert report["policy_id"] == "EPSB-1"
    assert report["source_contract"]["market_or_funding_rows_loaded"] == 0
    assert report["source_contract"]["source_quarantined_rows"] == 1
    assert report["opened_outcome_windows"] == []
    assert report["simulation_run"] is False
    assert report["policy"]["mutable_parameters"] == []
    assert report["source_only_distributions"]["stage1_2020_2022"] == {
        "trades": 37,
        "longs": 13,
        "shorts": 24,
        "max_single_month_count": 3,
        "max_single_month_share": 3 / 37,
    }


def test_frozen_preregistration_artifact_replays() -> None:
    stored = json.loads(Path(prereg.DEFAULT_OUTPUT).read_text())
    replay = prereg.build_report(Path(prereg.DEFAULT_CLOCK))
    assert replay == stored
