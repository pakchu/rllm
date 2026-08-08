import hashlib,json
from pathlib import Path
from training import preregister_options_crowding_deleveraging_relay_v3 as p
P=Path('results/options_crowding_deleveraging_relay_preregistration_v3_2026-08-08.json')
def test_v3_artifact_is_hash_bound_before_incidence():
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='b03ee6cfc2008d5ce277fb0b6bfa27ea6a944bf25bf695af903ce25ea86a21bd'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==p.v2.v1.canonical_hash(core)
 assert d['research_boundary']['v3_candidate_incidence_opened'] is False and d['outcomes_opened'] is False
