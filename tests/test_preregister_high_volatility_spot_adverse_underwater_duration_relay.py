import json
from training import preregister_high_volatility_spot_adverse_underwater_duration_relay as p
def test_spot_duration_is_frozen_before_incidence():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVSAUD-8";assert x["policy"]["duration_rank_max"]==.25;assert x["research_boundary"]["HVAUD_event_set_reused"] is False;assert x["research_boundary"]["exact_spot_duration_incidence_or_outcomes_known"] is False;json.dumps(x,allow_nan=False)
def test_variation_source_hash():assert p.sha256(p.VARIATION)==p.VARIATION_SHA
