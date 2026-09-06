import json
from training import preregister_high_volatility_caspc_ehlers_premium_activity_router as p
def test_premium_router_is_frozen_before_source():
 v=p.build();p.validate(v);assert v['policy_id']=='HVCELVPAR-8';assert v['source_incidence_opened'] is False;assert v['construction']['router'].startswith('relative activity rank<=0.50');assert v['construction']['rank'].startswith('strict-prior');assert v['research_boundary']['repair_of_prior_candidate'] is False;json.dumps(v,allow_nan=False)
