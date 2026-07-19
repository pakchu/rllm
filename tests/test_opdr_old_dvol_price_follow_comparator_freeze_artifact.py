from __future__ import annotations

import json
from pathlib import Path

from training.freeze_opdr_old_dvol_price_follow_comparator import _canonical_hash


ARTIFACT = Path(
    "results/opdr_old_dvol_price_follow_comparator_freeze_2026-07-19.json"
)


def test_frozen_comparator_artifact_is_sealed_and_self_consistent() -> None:
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == _canonical_hash(core)
    assert report["manifest_hash"] == (
        "52184ed4d7964b5c9a26dcaf3e433542ba629abad099f987731383ce3f826258"
    )
    assert report["opdr_outcomes_opened"] is False
    assert report["clock"] == {
        "path": "data/opdr_old_dvol_price_follow_comparator_2023h2.csv.gz",
        "sha256": (
            "a9c9d1c8d32510e63e604dfdc8b9d079f7e7a4bc206fd0a0197cad8c65b03d3d"
        ),
        "rows": 29,
        "first_entry": "2023-07-13 10:05:00",
        "last_exit": "2023-12-23 07:05:00",
    }
