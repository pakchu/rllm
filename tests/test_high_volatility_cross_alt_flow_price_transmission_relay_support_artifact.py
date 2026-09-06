import hashlib
import json
from pathlib import Path

from training import preregister_high_volatility_cross_alt_flow_price_transmission_relay as prereg

ARTIFACT = Path("results/high_volatility_cross_alt_flow_price_transmission_relay_support_2026-08-13.json")
EXPECTED_SHA = "8197e4735c2daaa14962df82660d0888c63db152bbad202199c0b060fab9bbc2"


def test_source_rejection_is_immutable_blind_and_terminal():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == EXPECTED_SHA
    value = json.loads(ARTIFACT.read_text())
    manifest_hash = value.pop("manifest_hash")
    assert prereg.canonical_hash(value) == manifest_hash
    assert value["policy_id"] == "HVCAFPT-8"
    assert value["support_passed"] is False
    assert value["decision"] == "terminal_source_support_reject"
    assert value["advance_to_gross9_novelty"] is False
    assert value["advance_to_economic_outcomes"] is False
    assert value["postentry_return_pnl_execution_price_opened"] is False
    assert value["funding_values_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert {name: split["events"] for name, split in value["support"].items()} == {
        "train": 18, "test": 49, "eval": 44, "final": 25
    }
    assert value["support_checks"]["train_side_balance"] is False
    assert value["support_checks"]["test_side_balance"] is False
    assert value["support_checks"]["eval_side_balance"] is False
