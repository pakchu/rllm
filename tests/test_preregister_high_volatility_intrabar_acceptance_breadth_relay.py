import json
from training import preregister_high_volatility_intrabar_acceptance_breadth_relay as p

def test_hviabr_is_outcome_blind_singleton():
 r=p.build();assert r["policy_id"]=="HVIABR-8";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert r["features"]["acceptance_breadth"]=="(count(close_location>0.5)-count(close_location<0.5))/valid_count; strict nonzero";assert r["policy"]["acceptance_strength_rank_min"]==.70;assert r["policy"]["variation_rank_min"]==.65;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["repair_of_prior_candidate"] is False;core={k:v for k,v in r.items() if k!="manifest_hash"};assert r["manifest_hash"]==p.canonical_hash(core);json.dumps(r,allow_nan=False)
