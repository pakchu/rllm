from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.preregister_post_funding_cross_sectional_crowding_release import (
    canonical_hash,
)


RESULT = Path(
    "results/post_funding_cross_sectional_crowding_release_support_2026-07-17.json"
)
CLOCK = Path("data/post_funding_cross_sectional_crowding_release_clock_2023_2024.csv.gz")
EXPECTED_RESULT_SHA256 = "36d8f9078e095be9948524147af29e2a31929906223d53e3f10498a1e91e1ae7"
EXPECTED_CLOCK_SHA256 = "dcec37a6ab977e1312202424bedec3105de055af6541cda80ea90d1db1a8f28f"


def test_pfcr1_support_artifact_is_locked_and_rejected_before_outcomes() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_RESULT_SHA256
    assert hashlib.sha256(CLOCK.read_bytes()).hexdigest() == EXPECTED_CLOCK_SHA256
    payload = json.loads(RESULT.read_text())
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    assert canonical_hash(body) == payload["manifest_hash"]
    assert payload["post_entry_returns_calculated"] is False
    assert payload["2023_fit_opened"] is False
    assert payload["2024_test_opened"] is False
    assert payload["support"]["passes_support"] is False
    assert (
        payload["support"]["gates"]["maximum_month_share_at_most_0_20"]
        is False
    )


def test_pfcr1_support_had_breadth_but_excess_month_concentration() -> None:
    payload = json.loads(RESULT.read_text())
    support = payload["support"]
    assert support["events"] == 177
    assert support["year_counts"] == {"2023": 86, "2024": 91}
    assert support["unique_ordered_pairs"] == 23
    assert support["maximum_ordered_pair_share"] <= 0.25
    assert support["maximum_month_share"] > 0.20
