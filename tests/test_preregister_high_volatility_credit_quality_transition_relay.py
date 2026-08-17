import json
from training import preregister_high_volatility_credit_quality_transition_relay as p
def test_transition_is_frozen_before_incidence():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVCQTR-24";assert x["features"]["transition"].endswith("opposite signs");assert x["research_boundary"]["HVCREDIT_event_set_or_controls_reused"] is False;assert x["research_boundary"]["exact_transition_incidence_or_outcomes_known"] is False;json.dumps(x,allow_nan=False)
def test_source_hash():assert p.sha256(p.SOURCE)==p.SOURCE_SHA
