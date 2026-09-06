from training import preregister_high_volatility_small_ticket_exhaustion_reversal as prereg

def test_singleton_outcome_blind_contract():
 p=prereg.build();assert p['policy_id']=='HVSTER-8';assert p['singleton'];assert not p['outcomes_opened'];assert not p['source_incidence_opened'];assert not p['research_boundary']['grid'];assert not p['research_boundary']['repair_of_prior_candidate']
def test_frozen_rule_and_gates():
 p=prereg.build();f=p['features'];assert 'execution_count>=q75' in f['eligibility'];assert 'average_ticket<=q35' in f['eligibility'];assert p['clock']['hold']=='8 elapsed hours';assert p['economic_gates']['cagr_to_strict_mdd_min']==3.;assert p['economic_gates']['stress_cagr_to_strict_mdd_min']==2.5
def test_manifest_replays():
 p=prereg.build();h=p.pop('manifest_hash');assert prereg.canonical_hash(p)==h
