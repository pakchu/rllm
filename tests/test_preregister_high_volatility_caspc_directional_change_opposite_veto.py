import json
from training import preregister_high_volatility_caspc_directional_change_opposite_veto as p
def test_directional_change_veto_frozen():
 v=p.build();p.validate(v);assert v['policy_id']=='HVCASPCDCV-8';assert v['construction']['veto'].startswith('active opposite HVDCS side emits cash');assert v['source_incidence_opened'] is False;assert v['research_boundary']['repair_of_prior_candidate'] is False;json.dumps(v,allow_nan=False)
