import json

from training import build_high_volatility_thursday_macro_reaction_fade_support as support


def test_support_artifact_is_terminal_and_keeps_later_gates_closed():
    payload = json.loads(support.RESULT.read_text())
    assert payload["policy_id"] == "HVTMRF-6"
    assert not payload["support_passed"]
    assert not payload["advance_to_gross9_novelty"]
    assert not payload["advance_to_economic_outcomes"]
    assert not payload["gross9_rows_opened"]
    assert payload["decision"] == "terminal_source_support_reject"
    assert [payload["support"][name]["events"] for name in support.SPLITS] == [3, 23, 14, 9]
    manifest_hash = payload.pop("manifest_hash")
    assert support.prereg.canonical_hash(payload) == manifest_hash
