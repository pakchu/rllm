from __future__ import annotations

import json
from pathlib import Path

from training.run_sec_edgar_bitcoin_product_access_synthetic_gate import (
    canonical_hash,
    evaluate_records,
    sha256_file,
)


ARTIFACT = Path(
    "results/sec_edgar_bitcoin_product_access_synthetic_gate_2026-07-22.json"
)
ARTIFACT_SHA256 = "036af95ce032bdf9de2b10a742f457cdc09e6096b60616f5d5f5da5c4001e2c4"
EXPECTED_FAILURES = {
    "suspended_customer_trading",
    "terminated_client_custody",
    "delisted_retail_bitcoin",
    "regulatory_access_halt",
    "mou_and_pilot_only",
    "third_party_access",
    "mixed_access_direction",
}


def test_committed_synthetic_artifact_is_exact_and_self_consistent() -> None:
    assert sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    manifest = dict(payload)
    recorded_manifest = manifest.pop("manifest_hash")
    assert recorded_manifest == canonical_hash(manifest)
    runner = payload["anchors"]["runner"]
    assert sha256_file(runner["path"]) == runner["sha256"]
    assert payload["anchors"]["preregistration"]["sha256"] == (
        "ab975eea454fbe1a784adaee979c5ad6162be9b18363c7fe3aa47959e075b883"
    )


def test_artifact_reproduces_failed_semantic_gate_and_passed_memory_gate() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    evaluation = payload["evaluation"]
    assert evaluate_records(payload["records"], payload["runtime"]) == evaluation
    assert evaluation["passed"] is False
    assert evaluation["counts"] == {
        "actual_classes": {
            "BTC_ACCESS_EXPANSION": 9,
            "BTC_ACCESS_RETRACTION": 1,
            "PARSE_FAILURE": 5,
            "UNSUPPORTED": 9,
        },
        "cases": 24,
        "exact_expected_cases": 17,
        "guarded_cases": 2,
        "model_calls": 22,
        "parsed_model_outputs": 17,
        "quote_valid_model_outputs": 17,
    }
    failed = {
        row["name"]
        for row in payload["records"]
        if not row["expected_match"] or not row["parsed_ok"] or not row["quote_valid"]
    }
    assert failed == EXPECTED_FAILURES
    checks = evaluation["checks"]
    assert checks["all_expected_classes"] is False
    assert checks["all_model_outputs_parse"] is False
    assert checks["all_model_quotes_validate"] is False
    assert checks["peak_allocated_within_cap"] is True
    assert checks["peak_reserved_within_cap"] is True
    assert payload["runtime"]["peak_allocated_bytes"] == 6_856_160_256
    assert payload["runtime"]["peak_reserved_bytes"] == 6_893_338_624


def test_failure_keeps_bodies_outcomes_future_and_live_deployment_sealed() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["outcome_boundary"] == {
        "2024_or_later_source_rows_read": 0,
        "btc_market_rows_read": 0,
        "clean_room_claimed": False,
        "filing_bodies_opened": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "historical_semantic_rows_opened": 0,
        "return_or_pnl_fields_read": 0,
    }
    decision = payload["decision"]
    assert decision["synthetic_gate_passed"] is False
    assert decision["filing_body_transport_authorized"] is False
    assert decision["historical_semantic_execution_authorized"] is False
    assert decision["novelty_evaluation_authorized"] is False
    assert decision["economic_evaluation_authorized"] is False
    assert decision["2024_or_later_authorized"] is False
    assert decision["target_3060ti_live_deployment_authorized"] is False
