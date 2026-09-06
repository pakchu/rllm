import hashlib
import json
from pathlib import Path

from training import preregister_high_volatility_large_ticket_temporal_clustering_relay as prereg


RESULT = Path("results/high_volatility_large_ticket_temporal_clustering_relay_support_2026-08-13.json")
EXPECTED_SHA256 = "52271ac02bfef0429155947ea5c05a67aff47cdac9ebb3727015203a537300ce"


def test_source_pass_is_immutable_and_outcomes_remain_blind() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert prereg.canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVLTTC-8"
    assert payload["support_passed"] is True
    assert payload["decision"] == "pass_to_novelty"
    assert payload["advance_to_gross9_novelty"] is True
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert {name: value["events"] for name, value in payload["support"].items()} == {
        "train": 91,
        "test": 123,
        "eval": 147,
        "final": 86,
    }
    assert all(payload["support_checks"].values())
