import json

from training import evaluate_treasury_real_term_spread_relay_gross9_novelty as novelty


def test_gross9_novelty_pass_is_bound_and_outcomes_remain_sealed():
    payload = json.loads(novelty.OUTPUT.read_text())
    assert payload["policy_id"] == "TRTSR-24"
    assert payload["source_support_passed"] is True
    assert payload["every_gross9_sleeve_passed"] is True
    assert payload["gross9_novelty_status"] == "passed"
    assert payload["advance_to_economic_outcomes"] is True
    assert payload["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    assert payload["evidence_boundary"]["outcomes_opened"] is False
    assert all(record["passed"] for record in payload["gross9_sleeves"].values())
    manifest_hash = payload.pop("manifest_hash")
    assert novelty.canonical_hash(payload) == manifest_hash
