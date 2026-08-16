import json
from training import preregister_high_volatility_cross_structure_action_vote_oi_sponsorship as p
def test_hvcavois_is_frozen_before_source():
 v=p.build();p.validate(v);assert v['policy_id']=='HVCAVOIS-8';assert v['source_incidence_opened'] is False;assert v['construction']['gate'].startswith('strict log(OI_D/OI_D_minus_8h)>0');assert v['research_boundary']['repair_of_prior_candidate'] is False;json.dumps(v,allow_nan=False)
