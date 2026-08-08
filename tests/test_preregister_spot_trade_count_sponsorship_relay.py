from training import preregister_spot_trade_count_sponsorship_relay as prereg
def test_stcsr_is_singleton_outcome_blind_and_independent():
 r=prereg.build();prereg.validate(r);assert r["policy_id"]=="STCSR-12";assert r["singleton"] is True;assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_stcsr_count_share_clock_and_gates_are_frozen():
 r=prereg.build();assert "number_of_trades" in r["features"]["trade_counts"];assert "rank>=0.75" in r["features"]["spot_count_share_rank"];assert "rank>=0.65" in r["features"]["btc_variation_rank"];assert r["clock"]["entry"].startswith("exact 08:05 UTC");assert r["clock"]["hold"]=="12 elapsed hours";assert r["economic_gates"]["stop_on_first_failure"] is True
