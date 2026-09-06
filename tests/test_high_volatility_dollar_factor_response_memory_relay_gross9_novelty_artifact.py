import json

from training import evaluate_high_volatility_dollar_factor_response_memory_relay_gross9_novelty as novelty


def test_novelty_artifact_passes_every_sleeve_without_opening_economics():
    payload = json.loads(novelty.OUTPUT.read_text())
    assert payload["policy_id"] == "HVDFRM-12"
    assert payload["source_support_passed"]
    assert payload["every_gross9_sleeve_passed"]
    assert payload["advance_to_economic_outcomes"]
    assert all(item["passed"] for item in payload["gross9_sleeves"].values())
    boundary = payload["evidence_boundary"]
    assert boundary["economic_outcome_rows_opened"] == 0
    assert not boundary["portfolio_return_or_pnl_metrics_computed"]
    manifest_hash = payload.pop("manifest_hash")
    assert novelty.chash(payload) == manifest_hash
