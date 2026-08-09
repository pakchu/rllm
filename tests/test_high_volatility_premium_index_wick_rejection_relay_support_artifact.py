import json

from training import build_high_volatility_premium_index_wick_rejection_relay_support as support


def test_support_artifact_is_terminal_and_later_gates_remain_closed():
    payload = json.loads(support.RESULT.read_text())
    assert payload["policy_id"] == "HVPIWR-12"
    assert not payload["support_passed"]
    assert not payload["advance_to_gross9_novelty"]
    assert not payload["advance_to_economic_outcomes"]
    assert not payload["postentry_return_pnl_execution_price_opened"]
    assert not payload["gross9_rows_opened"]
    assert [payload["support"][name]["events"] for name in support.SPLITS] == [8, 19, 23, 24]
    assert payload["support"]["train"]["max_month_share"] == 0.5
    manifest_hash = payload.pop("manifest_hash")
    assert support.prereg.canonical_hash(payload) == manifest_hash
