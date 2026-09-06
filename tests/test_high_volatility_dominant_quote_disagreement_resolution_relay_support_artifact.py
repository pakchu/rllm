import json

from training import build_high_volatility_dominant_quote_disagreement_resolution_relay_support as support


def test_support_artifact_is_terminal_before_later_gates():
    payload = json.loads(support.RESULT.read_text())
    assert payload["policy_id"] == "HVDQDR-8"
    assert not payload["support_passed"]
    assert not payload["advance_to_gross9_novelty"]
    assert not payload["advance_to_economic_outcomes"]
    assert not payload["information_embargo_audit"]["postentry_return_pnl_execution_price_opened"]
    assert not payload["information_embargo_audit"]["gross9_rows_opened"]
    assert [payload["support"][name]["events"] for name in support.SPLITS] == [0, 2, 1, 1]
    manifest_hash = payload.pop("manifest_hash")
    assert support.prereg.canonical_hash(payload) == manifest_hash
