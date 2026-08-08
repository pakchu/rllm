import hashlib,json
from pathlib import Path
from training.preregister_options_crowding_deleveraging_relay import canonical_hash
P=Path('results/options_crowding_deleveraging_relay_source_support_v3_veto_2026-08-08.json')
def test_v3_veto_is_before_incidence_and_outcomes():
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='468fca3defbf5d516e8619c436dc9b9611b558a923f10a629e440bbc1fcdca67'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==canonical_hash(core)
 assert d['candidate_incidence_opened'] is False and d['economic_outcomes_opened'] is False
 assert d['decision']=='TERMINAL_SOURCE_SUPPORT_REJECT_NO_RETRY_UNDER_V3'
