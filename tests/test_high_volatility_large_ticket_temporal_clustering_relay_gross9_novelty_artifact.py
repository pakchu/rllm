import hashlib
import json
from pathlib import Path

from training import export_gross9_structural_clocks as gross9


RESULT = Path("results/high_volatility_large_ticket_temporal_clustering_relay_gross9_novelty_2026-08-13.json")
EXPECTED_SHA256 = "d7701615cd82ddeae675a3bec63590ba3ab2402157476d691e8994e0ffae288e"


def test_novelty_pass_is_immutable_and_economically_blind() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert gross9.canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVLTTC-8"
    assert payload["source_support_passed"] is True
    assert payload["every_gross9_sleeve_passed"] is True
    assert payload["gross9_novelty_status"] == "passed"
    assert payload["advance_to_economic_outcomes"] is True
    boundary = payload["evidence_boundary"]
    assert boundary["btc_execution_rows_opened"] == 0
    assert boundary["btc_price_or_return_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert boundary["economic_outcome_rows_opened"] == 0
    assert boundary["outcomes_opened"] is False
    for result in payload["gross9_sleeves"].values():
        assert result["passed"] is True
        assert all(result["checks"].values())
