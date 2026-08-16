import json
from training import preregister_high_volatility_caspc_ehlers_funding_disagreement as p
def test_funding_disagreement_is_frozen_before_source():
 v=p.build();p.validate(v);assert v["policy_id"]=="HVCELVFD-8";assert v["source_incidence_opened"] is False;assert v["outcomes_opened"] is False;assert v["construction"]["gate"]=="strict side*funding_rate<0; zero or missing is ineligible";assert v["research_boundary"]["repair_of_prior_candidate"] is False;json.dumps(v,allow_nan=False)
