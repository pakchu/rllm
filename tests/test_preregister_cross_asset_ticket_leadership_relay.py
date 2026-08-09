from training import preregister_cross_asset_ticket_leadership_relay as prereg
def test_outcome_blind_singleton():
 p=prereg.build();assert p['policy_id']=='CATLR-12';assert p['singleton'];assert not p['outcomes_opened'];assert not p['source_incidence_opened'];assert not p['research_boundary']['grid'];assert not p['research_boundary']['repair_of_prior_candidate']
def test_cross_sectional_ticket_rule_is_fixed():
 f=prereg.build()['features'];assert len(f['symbols'])==7;assert 'cross-sectional median of six alt' in f['ticket_leadership'];assert 'rank>=0.75' in f['leadership_rank'];assert prereg.build()['clock']['hold']=='12 elapsed hours'
def test_manifest_replays():
 p=prereg.build();h=p.pop('manifest_hash');assert prereg.canonical_hash(p)==h
