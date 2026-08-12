import hashlib
import json
from pathlib import Path

from training import build_high_volatility_safehaven_relative_carry_dislocation_relay_support as support


RESULT = Path("results/high_volatility_safehaven_relative_carry_dislocation_relay_support_2026-08-13.json")
EXPECTED_SHA256 = "50ae3e7be27ab340f9d81f0633cd07aa580b30a205bff1a83ae8f5fbcd7ae92e"


def test_terminal_source_rejection_is_immutable_and_outcomes_remain_sealed() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert support.chash(payload) == manifest_hash
    assert payload["policy_id"] == "HVSCRD-8"
    assert payload["support_passed"] is False
    assert payload["decision"] == "terminal_source_support_reject"
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert {name: value["events"] for name, value in payload["support"].items()} == {
        "train": 13,
        "test": 25,
        "eval": 30,
        "final": 16,
    }
    assert payload["support"]["train"]["max_month_share"] == 6 / 13
    assert payload["clock"]["rows"] == 84
