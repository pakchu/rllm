from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import build_cross_collateral_inventory_pressure_absorption_support as s


SUPPORT = Path(
    "results/cross_collateral_inventory_pressure_absorption_support_2026-07-19.json"
)
CLOCK = Path(
    "data/cross_collateral_inventory_pressure_absorption_clock_2021_2023.csv.gz"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_support_rejects_without_outcomes() -> None:
    report = json.loads(SUPPORT.read_text(encoding="utf-8"))
    assert _sha256(SUPPORT) == "2866ed8e87a9c51901baf7120a428b51c71517c3a803c5a39a4931132998302f"
    assert _sha256(CLOCK) == "a96d06ecda35fd7f0f75a8015ab907e280c4d4b8c06620a9da3d874adb6523f9"
    assert report["manifest_hash"] == s._canonical_hash(
        {key: value for key, value in report.items() if key != "manifest_hash"}
    )
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["support_passed"] is False
    assert report["advance_to_train_outcomes"] is False
    assert report["failed_checks"] == ["2023_month_concentration"]
