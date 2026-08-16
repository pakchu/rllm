import json
from training import preregister_high_volatility_dcs_ticket_premium_activity_router as p
def test_dcs_ticket_premium_router_frozen():
 v=p.build();p.validate(v);assert v['policy_id']=='HVDCSATPAPAR-8';assert v['construction']['router'].startswith('relative activity rank<=0.50');assert v['source_incidence_opened'] is False;assert v['research_boundary']['repair_of_prior_candidate'] is False;json.dumps(v,allow_nan=False)
