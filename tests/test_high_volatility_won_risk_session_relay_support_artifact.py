import hashlib
import json
from pathlib import Path

from training import preregister_high_volatility_won_risk_session_relay as prereg


RESULT = Path("results/high_volatility_won_risk_session_relay_support_2026-08-13.json")
EXPECTED_SHA256 = "3966945f979e604e28b5b120a98705cadca6da581ad7c6979fab5198b0b8d55a"


def test_terminal_source_rejection_is_immutable_and_blind() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert prereg.canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVWRSR-8"
    assert payload["support_passed"] is False
    assert payload["decision"] == "terminal_source_support_reject"
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert {name: value["events"] for name, value in payload["support"].items()} == {
        "train": 12,
        "test": 19,
        "eval": 26,
        "final": 12,
    }
    assert payload["support"]["train"]["max_month_share"] == 0.5
    assert payload["support"]["eval"]["minority_side_share"] < 0.2
    assert payload["support"]["final"]["max_month_share"] == 0.5
