from training import preregister_options_crowding_deleveraging_relay_v3 as p

def test_v3_uses_raw_asof_availability_without_mechanism_change():
 d=p.build();assert d['policy']['policy_id']=='OCDR-12B';assert d['policy']['oi_asof_max_age_minutes']==5
 assert 'without floor, round, snap or fill' in d['causal_clock']['oi_archive_availability']
 assert d['research_boundary']['mechanism_threshold_side_hold_changed_from_v2'] is False
 assert d['research_boundary']['v3_candidate_incidence_opened'] is False
