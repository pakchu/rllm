import json
from training import preregister_high_volatility_har_variation_surprise_relay as p
def test_hvhvs_is_outcome_blind_singleton():
 r=p.build();assert r['policy_id']=='HVHVS-8';assert r['outcomes_opened'] is False;assert r['source_incidence_opened'] is False;assert r['gross9_rows_opened'] is False;assert r['research_boundary']['candidate_count']==1;assert r['policy']['har_weekly_blocks']==21;assert r['policy']['surprise_rank_min']==.75;core={k:v for k,v in r.items() if k!='manifest_hash'};assert r['manifest_hash']==p.canonical_hash(core);json.dumps(r,allow_nan=False)
