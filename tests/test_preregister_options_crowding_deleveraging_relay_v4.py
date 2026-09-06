from training import preregister_options_crowding_deleveraging_relay_v4 as p
def test_v4_adds_only_frozen_exact_funding_mark_authority():
 d=p.build();assert d['policy']['policy_id']=='OCDR-12C';assert d['research_boundary']['mechanism_threshold_side_hold_changed_from_v3'] is False
 assert d['economic_gates']['funding_mark_missing_action']=='terminal failure';assert d['research_boundary']['v4_candidate_incidence_opened'] is False
