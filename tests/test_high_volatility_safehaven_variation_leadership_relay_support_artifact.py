import hashlib
import json
from pathlib import Path

from training import build_high_volatility_safehaven_variation_leadership_relay_support as support


RESULT = Path("results/high_volatility_safehaven_variation_leadership_relay_support_2026-08-13.json")
EXPECTED_SHA256 = "61284a26c618585a99b341c67aa1eedeb89c948719a3379e17eb930031793682"


def test_terminal_source_rejection_is_immutable_and_outcomes_remain_sealed() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert support.chash(payload) == manifest_hash
    assert payload["policy_id"] == "HVSVL-8"
    assert payload["support_passed"] is False
    assert payload["decision"] == "terminal_source_support_reject"
    assert not payload["advance_to_gross9_novelty"]
    assert not payload["advance_to_economic_outcomes"]
    assert not payload["postentry_return_pnl_execution_price_opened"]
    assert not payload["gross9_rows_opened"]
    assert {name: value["events"] for name, value in payload["support"].items()} == {"train": 21, "test": 28, "eval": 27, "final": 16}
    assert payload["support"]["train"]["max_month_share"] == 11 / 21
    assert payload["clock"]["rows"] == 92
