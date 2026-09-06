import hashlib, json
from pathlib import Path

from training import preregister_high_volatility_daily_quarter_opening_flow_surprise_relay as p

RESULT = Path("results/high_volatility_daily_quarter_opening_flow_surprise_relay_support_2026-08-13.json")
EXPECTED = "f01ed50c275815d4cf881b802b5d123c875b7c54bd7dc52d9e527da7a67779ad"


def test_source_rejection_is_immutable_blind_and_terminal():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    result = json.loads(RESULT.read_text())
    manifest_hash = result.pop("manifest_hash")
    assert p.canonical_hash(result) == manifest_hash
    assert result["policy_id"] == "HVDQOFS-12"
    assert result["support_passed"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert result["support"]["train"]["max_month_share"] > 0.45
    assert not result["advance_to_gross9_novelty"]
    assert not result["advance_to_economic_outcomes"]
    assert not result["postentry_return_pnl_execution_price_opened"]
    assert not result["funding_values_opened"]
    assert not result["gross9_rows_opened"]
