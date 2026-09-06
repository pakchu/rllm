from training import preregister_korean_cash_leadership_relay as prereg
def test_kclr_is_singleton_outcome_blind_and_independent():
 r=prereg.build();prereg.validate(r);assert r["policy_id"]=="KCLR-12";assert r["singleton"] is True;assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_kclr_leadership_clock_and_gates_are_frozen():
 r=prereg.build();assert "abs(Upbit return)>abs(Binance" in r["features"]["leadership"];assert "rank>=0.65" in r["features"]["btc_variation_rank"];assert r["clock"]["entry"].startswith("exact 08:05 UTC");assert r["clock"]["hold"]=="12 elapsed hours";assert r["economic_gates"]["stop_on_first_failure"] is True
