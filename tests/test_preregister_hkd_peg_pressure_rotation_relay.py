from training import preregister_hkd_peg_pressure_rotation_relay as prereg

def test_hpprr_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r["policy_id"]=="HPPRR-12";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False

def test_hpprr_freezes_peg_pressure_and_volatile_regime():
 r=prereg.build();p=r["policy"];assert p["pressure_magnitude_rank_min"]==.70;assert p["realized_variation_rank_min"]==.65;assert p["hold_hours"]==12;assert r["source_plan"]["fx"]["symbol"]=="USDHKD";assert r["clock"]["side"]=="negative strict sign of peg_pressure_change"

def test_hpprr_controls_cannot_be_promoted():
 r=prereg.build();assert r["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True;assert r["research_boundary"]["promoted_prior_control"] is False;assert r["research_boundary"]["usdhkd_values_used_to_select_rule"] is False

def test_hpprr_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
