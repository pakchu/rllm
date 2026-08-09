import json

from training import evaluate_high_volatility_liquidity_impact_relay_gross9_novelty as novelty


def test_terminal_novelty_rejection_is_hash_bound():
    payload = json.loads(novelty.OUTPUT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == novelty.canonical_hash(core)
    assert payload["gross9_novelty_status"] == "failed"
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["evidence_boundary"]["outcomes_opened"] is False
    assert payload["evidence_boundary"]["economic_outcome_rows_opened"] == 0


def test_only_near_six_hour_gate_failed():
    payload = json.loads(novelty.OUTPUT.read_text())
    for result in payload["gross9_sleeves"].values():
        failed = {name for name, passed in result["checks"].items() if not passed}
        assert failed == {"one_to_one_6h_max_matched_share"}
