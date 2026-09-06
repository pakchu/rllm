import json

from training import evaluate_high_volatility_dollar_factor_response_memory_relay_economics as economics


def test_train_economics_is_terminal_and_does_not_authorize_later_stages():
    payload = json.loads(economics.OUTPUTS["train"].read_text())
    assert payload["policy_id"] == "HVDFRM-12"
    assert payload["stage"] == "train"
    assert not payload["passed"]
    assert not payload["advance_to_next_stage"]
    assert not payload["advance_to_post_stage_volatility_audit"]
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["primary"]["base"]["absolute_return_pct"] < 0
    assert payload["primary"]["base"]["mean_gross_underlying_bp"] < 20
    assert payload["primary"]["cluster_signflip"]["pvalue"] > .1
    assert not payload["later_stage_outcomes_opened"]
    manifest_hash = payload.pop("manifest_hash")
    assert economics.canonical_hash(payload) == manifest_hash
