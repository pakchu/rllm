import json
from training import preregister_cross_venue_volatility_ignition_breakout_relay as prereg

def test_cvvib_preregistration_is_outcome_blind_and_hash_bound():
    payload = prereg.build(); core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["policy_id"] == "CVVIB-6"
    assert payload["research_boundary"]["cvvib_candidate_incidence_opened"] is False
    assert payload["research_boundary"]["cvvib_post_entry_return_or_pnl_opened"] is False
    assert payload["research_boundary"]["grid"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False
    assert payload["policy"]["block_hours"] == 4

def test_written_cvvib_preregistration_matches_builder():
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
