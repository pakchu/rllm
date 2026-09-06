import json
from training import preregister_high_volatility_random_walk_index_cross_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVRWI-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert (x["policy"]["minimum_periods"],x["policy"]["maximum_periods"])==(1,14) and x["research_boundary"]["grid"] is False and x["research_boundary"]["repair_of_prior_candidate"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core);assert json.loads(p.DEFAULT_OUTPUT.read_text())==x
 assert p.canonical_hash({"한글":"RWI"})
