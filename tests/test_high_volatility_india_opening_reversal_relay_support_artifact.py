import hashlib
import json
from pathlib import Path

from training import build_high_volatility_india_opening_reversal_relay_support as support


RESULT = Path("results/high_volatility_india_opening_reversal_relay_support_2026-08-13.json")
EXPECTED_SHA256 = "34283a9258d7588d17926da5bc36112b3792df3729f48ddfb6a5230055be9566"


def test_terminal_source_rejection_is_immutable_and_outcomes_remain_sealed() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert support.chash(payload) == manifest_hash
    assert payload["policy_id"] == "HVINOR-8"
    assert payload["support_passed"] is False
    assert payload["decision"] == "terminal_source_support_reject"
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert {name: value["events"] for name, value in payload["support"].items()} == {
        "train": 0,
        "test": 0,
        "eval": 0,
        "final": 0,
    }
    assert payload["clock"]["rows"] == 0
