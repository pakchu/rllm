import hashlib,json
from pathlib import Path
from training.preregister_options_crowding_deleveraging_relay import canonical_hash

P=Path('results/options_crowding_deleveraging_relay_source_support_v2_veto_2026-08-08.json')

def test_v2_veto_preserves_no_incidence_or_outcome_boundary():
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='e45f993bc1fcaead3058fc0c31621cc69964c7aa14cf74f78960fb5159e43d97'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==canonical_hash(core)
 assert d['candidate_incidence_opened'] is False and d['economic_outcomes_opened'] is False
 assert d['decision']=='TERMINAL_SOURCE_SUPPORT_REJECT_NO_RETRY_UNDER_V2'
