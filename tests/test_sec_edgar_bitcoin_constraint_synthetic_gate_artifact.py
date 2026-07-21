from __future__ import annotations

import json
from pathlib import Path

from training.preregister_sec_edgar_bitcoin_constraint_transition_breadth import (
    canonical_hash,
    sha256_file,
)


ARTIFACT = Path("results/sec_edgar_bitcoin_constraint_synthetic_gate_2026-07-21.json")
ARTIFACT_SHA256 = "04e0e032531f95761fe63b24454a763b09e5c6f9a7d3b4ace6f88ac6fa2a14f8"


def test_committed_synthetic_failure_is_bound_and_opens_no_historical_data() -> None:
    assert sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    manifest = payload.pop("manifest_hash")
    assert canonical_hash(payload) == manifest
    runner = payload["anchors"]["runner"]
    assert sha256_file(runner["path"]) == runner["sha256"]
    assert payload["evaluation"]["passed"] is False
    assert payload["evaluation"]["counts"] == {
        "cases": 17,
        "model_calls": 15,
        "guarded_cases": 2,
        "parsed_model_outputs": 12,
        "exact_expected_cases": 14,
        "quote_valid_model_outputs": 12,
        "actual_labels": {
            "BTC_CONSTRAINT_BUFFER": 6,
            "BTC_CONSTRAINT_DRAW": 2,
            "PARSE_FAILURE": 3,
            "UNSUPPORTED": 6,
        },
    }
    failures = {
        row["name"]: row
        for row in payload["records"]
        if not row["expected_match"]
    }
    assert set(failures) == {
        "completed_sale",
        "pledged_collateral",
        "mixed_draw_buffer",
    }
    assert all(row["actual_label"] == "PARSE_FAILURE" for row in failures.values())
    assert payload["runtime"]["peak_allocated_bytes"] > 8 * 1024**3
    assert payload["outcome_boundary"] == {
        "filing_bodies_opened": 0,
        "historical_semantic_rows_opened": 0,
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "return_or_pnl_fields_read": 0,
        "2024_or_later_source_rows_read": 0,
        "clean_room_claimed": False,
    }
    decision = payload["decision"]
    assert decision["synthetic_gate_passed"] is False
    assert decision["filing_body_transport_authorized"] is False
    assert decision["historical_semantic_execution_authorized"] is False
    assert decision["economic_evaluation_authorized"] is False
    assert decision["2024_or_later_authorized"] is False
