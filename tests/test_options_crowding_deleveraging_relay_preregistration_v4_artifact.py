import hashlib,json
from pathlib import Path
from training import preregister_options_crowding_deleveraging_relay_v4 as p
P=Path('results/options_crowding_deleveraging_relay_preregistration_v4_2026-08-08.json')
def test_v4_artifact_is_frozen_before_incidence():
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='b3af208947fcda5fafc00b94726822849468664f212e21b4bf6f2b93f6c5b6b5'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==p.v3.v2.v1.canonical_hash(core)
 assert d['research_boundary']['v4_candidate_incidence_opened'] is False and d['outcomes_opened'] is False
