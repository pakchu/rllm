import json
from training import preregister_high_volatility_caspc_ehlers_oi_joint_sponsorship as p
def test_joint_candidate_is_frozen_before_incidence():
 v=p.build();p.validate(v)
 assert v["policy_id"]=="HVCELVOIS-8"
 assert v["combined_incidence_opened"] is False and v["combined_outcomes_opened"] is False
 assert v["component_ids"]==["HVCELV-8","HVCASPCOIS-8"]
 assert v["construction"]["timestamp_tolerance"]=="none"
 assert v["research_boundary"]["repair_of_prior_candidate"] is False
 json.dumps(v,allow_nan=False)
