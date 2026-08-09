from training import preregister_treasury_parallel_yield_shock_relay as prereg
def test_outcome_blind_singleton():
 p=prereg.build();assert p['policy_id']=='TPYSR-24';assert p['singleton'];assert not p['outcomes_opened'];assert not p['source_incidence_opened'];assert not p['research_boundary']['grid'];assert not p['research_boundary']['repair_of_prior_candidate']
def test_duration_factor_and_availability_are_fixed():
 f=prereg.build()['features'];assert 'D+1 00:00 UTC' in f['availability'];assert 'delta2' in f['yield_changes'];assert 'delta10' in f['yield_changes'];assert 'rank>=0.70' in f['factor_rank'];assert prereg.build()['clock']['hold']=='24 elapsed hours'
def test_manifest_replays():
 p=prereg.build();h=p.pop('manifest_hash');assert prereg.canonical_hash(p)==h
