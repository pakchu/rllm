from training import preregister_fear_greed_extremity_reversal as prereg
def test_fger_is_singleton_and_outcome_blind():
 r=prereg.build();prereg.validate(r);assert r["policy_id"]=="FGER-24";assert r["singleton"] is True;assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False
def test_fger_extremes_lag_clock_and_gates_are_frozen():
 r=prereg.build();assert r["features"]["extremes"]=={"fear_max":25,"greed_min":75};assert "D+1 00:00 UTC" in r["features"]["availability"];assert "rank>=0.65" in r["features"]["btc_variation_rank"];assert r["clock"]["hold"]=="24 elapsed hours";assert r["novelty_gates"]["candidate_near_6h_share_max"]==.35;assert r["economic_gates"]["stop_on_first_failure"] is True
