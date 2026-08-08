from training import preregister_funding_mark_gap_reconciliation_relay as prereg


def test_fmgrr_is_singleton_outcome_blind_and_independent():
    registration = prereg.build()
    prereg.validate(registration)
    assert registration["policy_id"] == "FMGRR-6"
    assert registration["singleton"] is True
    assert registration["outcomes_opened"] is False
    assert registration["source_incidence_opened"] is False
    assert registration["research_boundary"]["candidate_count"] == 1
    assert registration["research_boundary"]["grid"] is False
    assert registration["research_boundary"]["repair_of_prior_candidate"] is False


def test_fmgrr_mark_gap_clock_and_gates_are_frozen():
    registration = prereg.build()
    assert "mark price / 07:59" in registration["mechanism"]["side"]
    assert "rank>=0.75" in registration["features"]["absolute_gap_rank"]
    assert "rank>=0.65" in registration["features"]["btc_variation_rank"]
    assert registration["clock"]["entry"].startswith("exact 08:05 UTC")
    assert registration["clock"]["hold"] == "6 elapsed hours"
    assert registration["economic_gates"]["stop_on_first_failure"] is True
