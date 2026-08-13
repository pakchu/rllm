import hashlib
import json

from training import evaluate_high_volatility_exchange_reserve_pressure_relay_gross9_novelty as n


def test_hvexrp_gross9_pass_is_reproducible_and_outcome_blind():
    result = json.loads(n.OUTPUT.read_text(encoding="utf-8"))
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == n.canonical_hash(core)
    assert result["policy_id"] == "HVEXRP-24"
    assert result["every_gross9_sleeve_passed"] is True
    assert result["gross9_novelty_status"] == "passed"
    assert result["advance_to_economic_outcomes"] is True
    assert result["evidence_boundary"]["outcomes_opened"] is False
    assert result["evidence_boundary"]["btc_execution_rows_opened"] == 0
    assert result["evidence_boundary"]["funding_rows_opened"] == 0
    assert all(item["passed"] for item in result["gross9_sleeves"].values())
    assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest() == (
        "3d28ac63803b36a277b28b520d2e636c5ba7addb47d23999f48d0d958f8b7cc1"
    )
