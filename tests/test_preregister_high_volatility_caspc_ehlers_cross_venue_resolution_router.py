import json
from training import preregister_high_volatility_caspc_ehlers_cross_venue_resolution_router as p
def test_cross_venue_router_frozen():
 v=p.build();p.validate(v);assert v['policy_id']=='HVCELVCVDR-8';assert v['construction']['router'].startswith('active CVDR side overrides');assert v['source_incidence_opened'] is False;assert v['research_boundary']['repair_of_prior_candidate'] is False;json.dumps(v,allow_nan=False)
