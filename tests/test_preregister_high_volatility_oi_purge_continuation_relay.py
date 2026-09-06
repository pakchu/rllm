import json

from training import preregister_high_volatility_oi_purge_continuation_relay as prereg


def test_hvopcr_preregistration_is_outcome_blind_hash_bound_and_single_policy():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["policy_id"] == "HVOPCR-6"
    boundary = payload["research_boundary"]
    assert boundary["hvopcr_candidate_incidence_opened"] is False
    assert boundary["hvopcr_post_entry_return_or_pnl_opened"] is False
    assert boundary["candidate_count"] == 1
    assert boundary["grid"] is False
    assert boundary["repair_of_prior_candidate"] is False
    assert boundary["promoted_prior_control"] is False


def test_hvopcr_is_falling_oi_continuation_and_controls_cannot_be_promoted():
    payload = prereg.build()
    assert "strictly negative" in payload["clock"]["oi"]
    assert payload["clock"]["side"] == "sign of completed-hour BTC return"
    assert payload["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True
    assert payload["research_boundary"]["related_candidate_outcomes_used_to_define_hvopcr"] is False


def test_written_hvopcr_preregistration_matches_builder():
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
