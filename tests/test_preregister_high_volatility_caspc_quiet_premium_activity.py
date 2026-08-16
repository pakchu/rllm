import json
from training import preregister_high_volatility_caspc_quiet_premium_activity as p
def test_quiet_premium_gate_is_frozen_before_source():
 v=p.build();p.validate(v);assert v['policy_id']=='HVCASPCPQA-8';assert v['construction']['gate'].startswith('relative activity rank<=0.50');assert v['source_incidence_opened'] is False;assert v['research_boundary']['repair_of_prior_candidate'] is False;json.dumps(v,allow_nan=False)
