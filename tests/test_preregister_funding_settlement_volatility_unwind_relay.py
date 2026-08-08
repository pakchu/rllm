import json

from training import preregister_funding_settlement_volatility_unwind_relay as prereg


def test_fsvur_preregistration_is_outcome_blind_and_hash_bound():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["policy_id"] == "FSVUR-6"
    assert payload["research_boundary"]["fsvur_candidate_incidence_opened"] is False
    assert payload["research_boundary"]["fsvur_post_entry_return_or_pnl_opened"] is False
    assert payload["research_boundary"]["grid"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False
    assert payload["clock"]["event"].startswith("actual Binance")
    assert payload["clock"]["hold"] == "6 elapsed hours"


def test_written_fsvur_preregistration_matches_builder():
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
