from training import preregister_mexican_peso_risk_barometer_relay as prereg
def test_mxrbr_is_singleton_outcome_blind_and_independent():
 r=prereg.build();prereg.validate(r);assert r["policy_id"]=="MXRBR-12";assert r["singleton"] is True;assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_mxrbr_session_clock_and_gates_are_frozen():
 r=prereg.build();assert "[10:00,16:00)" in r["features"]["fx_session"];assert "negative sign" in r["mechanism"]["side"];assert "rank>=0.65" in r["features"]["btc_variation_rank"];assert r["clock"]["hold"]=="12 elapsed hours";assert r["novelty_gates"]["candidate_near_6h_share_max"]==.35;assert r["economic_gates"]["stop_on_first_failure"] is True
