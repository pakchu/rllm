import json

from training import evaluate_treasury_real_term_spread_relay_economics as economics


def test_train_rejection_is_bound_and_later_stages_remain_sealed():
    payload = json.loads(economics.OUTPUTS["train"].read_text())
    assert payload["policy_id"] == "TRTSR-24"
    assert payload["stage"] == "train"
    assert payload["passed"] is False
    assert payload["advance_to_next_stage"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["later_stage_outcomes_opened"] is False
    assert payload["checks"]["absolute_return_positive"] is True
    assert payload["checks"]["cagr_to_strict_mdd_min_3"] is False
    assert payload["checks"]["cluster_signflip_p_max_0_1"] is False
    assert payload["checks"]["stress_cagr_to_strict_mdd_min_2_5"] is False
    assert payload["checks"]["each_calendar_half_positive"] is False
    manifest_hash = payload.pop("manifest_hash")
    assert economics.canonical_hash(payload) == manifest_hash
