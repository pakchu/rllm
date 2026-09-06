import json

from training import build_fear_greed_persistence_diffusion_relay_support as support


def test_terminal_support_artifact_is_bound_and_outcomes_remain_sealed():
    payload = json.loads(support.RESULT.read_text())
    assert payload["policy_id"] == "FGPDR-24"
    assert payload["support_passed"] is False
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert [payload["support"][name]["events"] for name in support.SPLITS] == [0, 0, 0, 1]
    manifest_hash = payload.pop("manifest_hash")
    assert support.canonical_hash(payload) == manifest_hash
