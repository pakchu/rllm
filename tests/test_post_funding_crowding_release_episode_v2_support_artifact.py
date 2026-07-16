from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.preregister_post_funding_crowding_release_episode_v2 import canonical_hash


RESULT = Path("results/post_funding_crowding_release_episode_v2_support_2026-07-17.json")
CLOCK = Path("data/post_funding_crowding_release_episode_v2_clock_2023_2024.csv.gz")
EXPECTED_RESULT_SHA256 = "d09c3bc68efa541da00ab994ffac55f64cee1152c148361252e0206b53ffe083"
EXPECTED_CLOCK_SHA256 = "ebeb32ccaf1bc096c95f5c848ed34c6964d5be828555a8024a42a8f826586fbc"


def test_pfcr2_support_artifact_is_locked_before_outcomes() -> None:
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
    assert payload["support"]["passes_support"] is True


def test_pfcr2_support_is_broad_and_declustered() -> None:
    payload = json.loads(RESULT.read_text())
    support = payload["support"]
    assert payload["episode_cooldown_hours"] == 36
    assert support["events"] == 82
    assert support["year_counts"] == {"2023": 38, "2024": 44}
    assert support["unique_ordered_pairs"] == 20
    assert support["maximum_ordered_pair_share"] <= 0.25
    assert support["maximum_month_share"] <= 0.20
    assert len(support["long_symbols"]) == 6
    assert len(support["short_symbols"]) == 6
