from training import preregister_cboe_tail_pressure_crypto_volatility_transmission_relay as prereg
def test_ctptr_preregistration_is_outcome_blind_hash_bound_singleton():
 p=prereg.build();prereg.validate(p);core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==prereg.canonical_hash(core) and p["policy_id"]=="CTPTR-6" and p["outcomes_opened"] is False and p["source_incidence_opened"] is False and p["research_boundary"]["candidate_count"]==1 and p["research_boundary"]["grid"] is False and p["research_boundary"]["repair_of_prior_candidate"] is False
