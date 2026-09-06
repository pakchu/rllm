import json

from training import preregister_funding_polarity_volatility_ignition_relay as prereg


def test_fpvir_preregistration_is_outcome_blind_hash_bound_singleton():
    report = prereg.build()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == prereg.canonical_hash(core)
    assert report["policy_id"] == "FPVIR-6"
    assert report["outcomes_opened"] is False
    assert report["source_incidence_opened"] is False
    boundary = report["research_boundary"]
    assert boundary["fpvir_candidate_incidence_opened"] is False
    assert boundary["fpvir_post_entry_return_or_pnl_opened"] is False
    assert boundary["candidate_count"] == 1
    assert boundary["grid"] is False
    assert boundary["repair_of_prior_candidate"] is False
    assert boundary["promoted_prior_control"] is False
    assert report["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True


def test_written_fpvir_preregistration_matches_builder():
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
