import hashlib
import json
from pathlib import Path
from training import preregister_high_volatility_cross_alt_closing_impulse_dominance_relay as p

RESULT = Path("results/high_volatility_cross_alt_closing_impulse_dominance_relay_support_2026-08-13.json")
EXPECTED = "5660b7649d4ea28a8d2c02850eb0e920de0cc21ecbd1e8ec4030ca757aa41d54"


def test_terminal_source_rejection_is_immutable_and_blind():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    result = json.loads(RESULT.read_text())
    manifest_hash = result.pop("manifest_hash")
    assert p.canonical_hash(result) == manifest_hash
    assert result["policy_id"] == "HVCACID-8"
    assert not result["support_passed"]
    assert result["decision"] == "terminal_source_support_reject"
    assert not result["advance_to_gross9_novelty"] and not result["advance_to_economic_outcomes"]
    assert not result["postentry_return_pnl_execution_price_opened"]
    assert not result["funding_values_opened"] and not result["gross9_rows_opened"]
    assert all(item["events"] == 0 for item in result["support"].values())
