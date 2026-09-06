import json
from training import preregister_high_volatility_caspc_ehlers_cash_side_router as p
def test_cash_router_is_frozen_before_source():
 v=p.build();p.validate(v);assert v['policy_id']=='HVCELVCSR-8';assert v['construction']['router'].startswith('emit sign(cash_return)');assert v['source_incidence_opened'] is False;assert v['research_boundary']['repair_of_prior_candidate'] is False;json.dumps(v,allow_nan=False)
