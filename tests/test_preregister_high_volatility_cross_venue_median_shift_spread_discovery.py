import json
from training import preregister_high_volatility_cross_venue_median_shift_spread_discovery as p
def test_spread_discovery_is_frozen_before_incidence():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVCVMSSD-8";assert x["policy"]["spread_rank_min"]==.75;assert x["research_boundary"]["prior_event_sets_reused"] is False;assert x["research_boundary"]["exact_spread_onset_incidence_or_outcomes_known"] is False;json.dumps(x,allow_nan=False)
def test_sources_are_hash_pinned():assert p.sha256(p.PERP)==p.PERP_SHA and p.sha256(p.SPOT)==p.SPOT_SHA
