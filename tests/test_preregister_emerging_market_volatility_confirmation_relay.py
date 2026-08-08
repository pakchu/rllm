from training import preregister_emerging_market_volatility_confirmation_relay as prereg

def test_emvcr_is_singleton_outcome_blind_and_not_rvsbr_repair():
 r=prereg.build();prereg.validate(r);assert r["policy_id"]=="EMVCR-8";assert r["singleton"] is True;assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_rvsbr"] is False
def test_emvcr_clock_confirmation_and_gates_are_frozen():
 r=prereg.build();assert "absolute shock_z >= 0.75" in r["features"]["shock_gate"];assert "negative sign(shock)" in r["features"]["confirmation"];assert "rank >= 0.65" in r["features"]["btc_variation_rank"];assert r["clock"]["hold"]=="8 elapsed hours";assert r["novelty_gates"]["candidate_near_6h_share_max"]==.35;assert r["economic_gates"]["stop_on_first_failure"] is True;assert r["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True
