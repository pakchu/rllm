from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import evaluate_venue_ticket_migration_shock as evaluate


RESULT = Path("results/venue_ticket_migration_shock_stage1_2020_2022_2026-07-17.json")
EXPECTED_SHA256 = "90a4a05e5a422a2641e2026a5cf68750709d62bac0fa41ff9a91ab40f9b709af"
EXPECTED_MANIFEST_HASH = (
    "ac26e8a61af6fa808c3e88dff0b37157c29f4b19acd4ef89a3a416d225f4b583"
)


def test_stage1_result_is_hash_valid_failed_and_keeps_2023_sealed() -> None:
    assert evaluate._sha256(RESULT) == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert evaluate._canonical_hash(core) == EXPECTED_MANIFEST_HASH
    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert payload["candidate_id"] == "VTMS-288"
    assert payload["stage"] == "stage1_2020_2022"
    assert payload["stage1_qualifies"] is False
    assert payload["next_action"] == "reject_keep_2023_and_2024plus_sealed"
    assert payload["sealed_after_run"] == ["2023", "2024", "2025", "2026_ytd"]
    assert payload["source"]["cutoff"] == evaluate.STAGE1_END.isoformat()
    assert payload["source"]["last_market_time"] == "2022-12-31 23:55:00"
    assert payload["source"]["last_funding_time"] == "2022-12-31 16:00:00"
    assert not evaluate.STAGE2_OUTPUT.exists()


def test_stage1_primary_statistics_and_failed_gates_are_frozen() -> None:
    payload = json.loads(RESULT.read_text())
    base = payload["base"]
    assert base["absolute_return_pct"] == pytest.approx(8.679967993601046)
    assert base["cagr_pct"] == pytest.approx(2.812775967715897)
    assert base["strict_mdd_pct"] == pytest.approx(24.34040684615497)
    assert base["cagr_to_strict_mdd"] == pytest.approx(0.11555994053403541)
    assert base["trade_count"] == 336
    assert base["weekly_cluster_signflip"]["p_value_one_sided"] == pytest.approx(
        0.3311766882331177
    )
    assert payload["stress_10bp"]["absolute_return_pct"] == pytest.approx(
        -4.998477593199368
    )
    failed = {name for name, passed in payload["gate"].items() if not passed}
    assert failed == {
        "cagr_to_strict_mdd_at_least_3",
        "strict_mdd_at_most_15pct",
        "weekly_cluster_p_at_most_0p10",
        "stress_absolute_return_positive",
    }
