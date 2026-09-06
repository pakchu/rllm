from training import preregister_high_volatility_causal_response_memory_relay as prereg
def test_outcome_blind_singleton():
 p=prereg.build();assert p['policy_id']=='HVCRMR-12';assert p['singleton'];assert not p['outcomes_opened'];assert not p['source_incidence_opened'];assert not p['research_boundary']['grid'];assert not p['research_boundary']['repair_of_prior_candidate']
def test_causal_memory_is_fixed():
 p=prereg.build()['features'];assert 'latest 32' in p['memory'];assert 'at least 16' in p['memory'];assert 'availability<=current decision' in p['memory'];assert 'U+12h+5m' in p['counterfactual_response']
def test_manifest_replays():
 p=prereg.build();h=p.pop('manifest_hash');assert prereg.canonical_hash(p)==h
