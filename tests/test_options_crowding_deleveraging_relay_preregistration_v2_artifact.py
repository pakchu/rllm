import hashlib,json
from pathlib import Path
from training import preregister_options_crowding_deleveraging_relay_v2 as p

P=Path('results/options_crowding_deleveraging_relay_preregistration_v2_2026-08-08.json')

def test_v2_artifact_is_hash_bound_before_incidence():
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='43d33daed35678c852e5cb02e908f3fa381f1accb08e72fd138b51238c168baf'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==p.v1.canonical_hash(core)
 assert d['research_boundary']['v2_candidate_incidence_opened'] is False
 assert d['outcomes_opened'] is False
