import json
from training import preregister_cross_asset_impact_isolation_continuation as p
def test_caiic_is_outcome_blind_singleton():
 r=p.build();assert r['policy_id']=='CAIIC-8';assert r['outcomes_opened'] is False;assert r['source_incidence_opened'] is False;assert r['gross9_rows_opened'] is False;assert r['research_boundary']['candidate_count']==1;assert r['policy']['impact_isolation_rank_min']==.9;core={k:v for k,v in r.items() if k!='manifest_hash'};assert r['manifest_hash']==p.canonical_hash(core);json.dumps(r,allow_nan=False)
