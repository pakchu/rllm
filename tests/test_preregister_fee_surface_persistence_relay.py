from training import preregister_fee_surface_persistence_relay as prereg
def test_outcome_blind_singleton():
 p=prereg.build();assert p['policy_id']=='FSPR-24';assert p['singleton'];assert not p['outcomes_opened'];assert not p['source_incidence_opened'];assert not p['research_boundary']['grid'];assert not p['research_boundary']['repair_of_prior_candidate']
def test_frozen_persistence_and_gates():
 p=prereg.build();assert 'at least four of five' in p['features']['broad_sign'];assert 'broad_sign[t-1]' in p['features']['persistence'];assert p['clock']['hold']=='24 elapsed hours';assert p['economic_gates']['cagr_to_strict_mdd_min']==3.
def test_manifest_replays():
 p=prereg.build();h=p.pop('manifest_hash');assert prereg.canonical_hash(p)==h
