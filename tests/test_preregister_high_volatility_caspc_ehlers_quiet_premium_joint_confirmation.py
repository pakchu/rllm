import json
from training import preregister_high_volatility_caspc_ehlers_quiet_premium_joint_confirmation as p

def test_frozen_joint_confirmation_contract():
 v=p.build();p.validate(v)
 assert v['policy_id']=='HVCELVPQA-8'
 assert v['component_ids']==['HVCELV-8','HVCASPCPQA-8']
 assert v['construction']['operator']=='exact-entry joint sponsorship intersection'
 assert v['combined_incidence_opened'] is False
 assert v['research_boundary']['repair_of_prior_candidate'] is False
 json.dumps(v,allow_nan=False)
