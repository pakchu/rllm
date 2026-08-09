import json
from training import preregister_cross_venue_efficiency_handoff_relay as prereg
def test_cveh_is_outcome_blind_singleton():
 r=prereg.build();assert r["policy_id"]=="CVEH-6";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["policy"]["spot_efficiency_rank_min"]==.8;assert r["policy"]["cash_handoff_rank_min"]==.75
 core={k:v for k,v in r.items() if k!="manifest_hash"};assert r["manifest_hash"]==prereg.canonical_hash(core);json.dumps(r,allow_nan=False)
