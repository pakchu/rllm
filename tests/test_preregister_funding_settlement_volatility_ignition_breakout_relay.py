import json

from training import preregister_funding_settlement_volatility_ignition_breakout_relay as prereg


def test_fsvibr_preregistration_is_outcome_blind_hash_bound_and_singleton():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["policy_id"] == "FSVIBR-6"
    boundary = payload["research_boundary"]
    assert boundary["fsvibr_candidate_incidence_opened"] is False
    assert boundary["fsvibr_post_entry_return_or_pnl_opened"] is False
    assert boundary["candidate_count"] == 1
    assert boundary["grid"] is False
    assert boundary["repair_of_prior_candidate"] is False
    assert boundary["promoted_prior_control"] is False


def test_fsvibr_uses_joint_ignition_and_nonpromotable_controls():
    payload = prereg.build()
    assert "both strictly positive" in payload["clock"]["volatility_ignition"]
    assert payload["clock"]["funding_rate"].startswith("event identity only")
    assert payload["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True


def test_written_fsvibr_preregistration_matches_builder():
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
