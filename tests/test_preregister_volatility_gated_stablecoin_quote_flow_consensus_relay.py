import json

from training import preregister_volatility_gated_stablecoin_quote_flow_consensus_relay as prereg


def test_vgsqf_preregistration_is_outcome_blind_hash_bound_and_single_policy():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["policy_id"] == "VGSQF-6"
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["grid"] is False


def test_vgsqf_boundary_prohibits_control_promotion_and_records_prior_knowledge():
    payload = prereg.build()
    boundary = payload["research_boundary"]
    assert boundary["prior_sqfd_train_outcome_known"] is True
    assert boundary["prior_sqfd_outcome_used_to_define_vgsqf"] is False
    assert boundary["prior_vgsfr_source_failure_known"] is True
    assert boundary["vgsfr_no_sequential_order_control_outcome_opened"] is False
    assert boundary["vgsqf_candidate_incidence_opened"] is False
    assert boundary["vgsqf_post_entry_return_or_pnl_opened"] is False
    assert boundary["repair_of_prior_candidate"] is False
    assert payload["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True


def test_written_vgsqf_preregistration_matches_builder():
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
