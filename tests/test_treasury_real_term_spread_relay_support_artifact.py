import json

from training import build_treasury_real_term_spread_relay_support as support


def test_support_pass_artifact_is_bound_and_outcomes_remain_sealed():
    payload = json.loads(support.RESULT.read_text())
    assert payload["policy_id"] == "TRTSR-24"
    assert payload["support_passed"] is True
    assert payload["advance_to_gross9_novelty"] is True
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert [payload["support"][name]["events"] for name in support.SPLITS] == [13, 32, 29, 15]
    assert all(payload["support_checks"].values())
    manifest_hash = payload.pop("manifest_hash")
    assert support.canonical_hash(payload) == manifest_hash


def test_official_real_yield_downloads_remain_hash_bound():
    payload = json.loads(support.SOURCE_MANIFEST.read_text())
    for record in payload["files"].values():
        assert support.sha(support.Path(record["path"])) == record["sha256"]
    manifest_hash = payload.pop("manifest_hash")
    assert support.canonical_hash(payload) == manifest_hash
