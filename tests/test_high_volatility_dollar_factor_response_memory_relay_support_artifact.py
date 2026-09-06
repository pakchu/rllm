import json

from training import build_high_volatility_dollar_factor_response_memory_relay_support as support


def test_support_artifact_passes_and_keeps_later_gates_closed():
    payload = json.loads(support.RESULT.read_text())
    assert payload["policy_id"] == "HVDFRM-12"
    assert payload["support_passed"]
    assert payload["advance_to_gross9_novelty"]
    assert not payload["advance_to_economic_metrics"]
    assert not payload["gross9_rows_opened"]
    assert [payload["support"][name]["events"] for name in support.SPLITS] == [41, 103, 89, 40]
    assert not payload["causal_audit"]["same_decision_label_used"]
    assert not payload["causal_audit"]["unmatured_label_used"]
    manifest_hash = payload.pop("manifest_hash")
    assert support.chash(payload) == manifest_hash
