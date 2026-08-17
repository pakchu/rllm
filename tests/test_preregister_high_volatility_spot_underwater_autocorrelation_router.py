import json
from training import preregister_high_volatility_spot_underwater_autocorrelation_router as p
def test_router_is_frozen_before_state_access():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVSAUDAR-8";assert x["policy"]["history_blocks"]==90;assert x["policy"]["minimum_history_blocks"]==60;assert x["state"]["current_excluded"] is True;assert x["research_boundary"]["exact_autocorrelation_routed_incidence_or_outcomes_known"] is False;json.dumps(x,allow_nan=False)
def test_sources_are_hash_pinned():
 assert p.sha256(p.STATE)==p.STATE_SHA
 for a in p.BASE.values():assert p.sha256(a["path"])==a["sha256"]
