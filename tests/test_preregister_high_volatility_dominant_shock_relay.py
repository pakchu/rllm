from training import preregister_high_volatility_dominant_shock_relay as prereg

def test_preregistration_is_hash_valid_and_outcome_blind():
 r=prereg.build();prereg.validate(r);assert r["policy_id"]=="HVDSR-12";assert r["source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert r["policy"]["hours"]==24;assert r["clock"]["hold"]=="12 elapsed hours"
def test_singleton_and_controls_frozen():
 r=prereg.build();assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["diagnostic_controls"]["cannot_be_promoted"] is True
