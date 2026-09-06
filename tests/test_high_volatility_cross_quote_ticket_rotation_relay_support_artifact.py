import json

from training import build_high_volatility_cross_quote_ticket_rotation_relay_support as support


def test_support_artifact_is_terminal_and_keeps_later_gates_closed():
    payload = json.loads(support.RESULT.read_text())
    assert payload["policy_id"] == "HVCQTR-8"
    assert not payload["support_passed"]
    assert not payload["advance_to_gross9_novelty"]
    assert not payload["advance_to_economic_outcomes"]
    assert not payload["information_embargo_audit"]["postentry_return_pnl_execution_price_opened"]
    assert not payload["information_embargo_audit"]["gross9_rows_opened"]
    assert payload["decision"] == "terminal_source_support_reject"
    assert [payload["support"][name]["events"] for name in support.SPLITS] == [5, 25, 34, 14]
    assert payload["support"]["train"]["max_month_share"] == 0.6
    manifest_hash = payload.pop("manifest_hash")
    assert support.prereg.canonical_hash(payload) == manifest_hash
