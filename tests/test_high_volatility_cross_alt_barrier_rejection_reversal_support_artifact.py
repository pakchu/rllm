import hashlib
import json
from pathlib import Path

from training import preregister_high_volatility_cross_alt_barrier_rejection_reversal as p


RESULT = Path("results/high_volatility_cross_alt_barrier_rejection_reversal_support_2026-08-13.json")
EXPECTED = "a378e06ce2deaf6fe4be03d98a1dd860e03a13122dfe49872a7b0cb712fa08c1"


def test_source_pass_is_immutable_blind_and_authorizes_only_novelty():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    result = json.loads(RESULT.read_text())
    manifest_hash = result.pop("manifest_hash")
    assert p.canonical_hash(result) == manifest_hash
    assert result["policy_id"] == "HVCABRR-8"
    assert result["support_passed"] and result["decision"] == "pass_to_novelty"
    assert result["advance_to_gross9_novelty"]
    assert not result["advance_to_economic_outcomes"]
    assert not result["postentry_return_pnl_execution_price_opened"]
    assert not result["funding_values_opened"] and not result["gross9_rows_opened"]
    assert {k: v["events"] for k, v in result["support"].items()} == {
        "train": 55,
        "test": 142,
        "eval": 166,
        "final": 79,
    }
    assert all(result["support_checks"].values())
