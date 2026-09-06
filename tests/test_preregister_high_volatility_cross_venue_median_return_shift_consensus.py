import json
from training import preregister_high_volatility_cross_venue_median_return_shift_consensus as p
def test_cross_venue_consensus_is_frozen_before_incidence():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVCVMRS-8";assert x["construction"]["additional_or_tuned_thresholds"]=="none";assert x["research_boundary"]["exact_cross_venue_consensus_incidence_or_outcomes_known"] is False;json.dumps(x,allow_nan=False)
def test_primary_is_hash_pinned():
 for a in p.PRIMARY.values():assert p.sha256(a["path"])==a["sha256"]
