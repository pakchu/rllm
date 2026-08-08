from training import preregister_scandinavian_haven_risk_rotation_relay as prereg

def test_shrr_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r["policy_id"]=="SHRR-12";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False

def test_shrr_freezes_risk_rotation_and_volatile_regime():
 r=prereg.build();p=r["policy"];assert p["risk_pressure_rank_min"]==.70;assert p["realized_variation_rank_min"]==.65;assert p["hold_hours"]==12;assert r["source_plan"]["fx"]["symbols"]==["USDSEK","USDCHF","USDJPY"];assert r["clock"]["side"]=="negative strict sign of risk_pressure"

def test_shrr_is_not_dollar_factor_or_prior_control_promotion():
 r=prereg.build();assert "opposite score orientations" in r["mechanism"]["why_distinct"];assert r["research_boundary"]["prior_fx_candidate_outcomes_used_to_set_shrr_rule"] is False;assert r["research_boundary"]["promoted_prior_control"] is False;assert r["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True

def test_shrr_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
