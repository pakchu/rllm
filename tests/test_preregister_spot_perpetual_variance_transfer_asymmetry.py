from training import preregister_spot_perpetual_variance_transfer_asymmetry as prereg

def test_spvta_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r['policy_id']=='SPVTA-8';assert r['outcomes_opened'] is False;assert r['source_incidence_opened'] is False;assert r['singleton'] is True;assert r['research_boundary']['candidate_count']==1;assert r['research_boundary']['grid'] is False;assert r['research_boundary']['repair_of_prior_candidate'] is False

def test_spvta_freezes_variance_relocation_and_volatile_regime():
 r=prereg.build();p=r['policy'];assert p['block_hours']==8;assert p['half_hours']==4;assert p['direction_hours']==2;assert p['relocation_rank_min']==.65;assert p['realized_variation_rank_min']==.65;assert p['hold_hours']==8;assert r['clock']['side']=='common strict final-two-hour return sign'

def test_spvta_is_not_prior_spot_control_promotion():
 r=prereg.build();assert r['research_boundary']['prior_spot_perpetual_event_sets_reused'] is False;assert r['research_boundary']['prior_candidate_outcomes_used_to_set_spvta_rule'] is False;assert r['research_boundary']['promoted_prior_control'] is False;assert r['diagnostic_controls']['diagnostic_controls_cannot_be_promoted'] is True

def test_spvta_hash_binds_core():
 r=prereg.build();assert r['manifest_hash']==prereg.canonical_hash({k:v for k,v in r.items() if k!='manifest_hash'})
