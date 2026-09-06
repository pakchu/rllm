import hashlib,json
from pathlib import Path
from training.build_options_crowding_deleveraging_relay_support_v4 import canonical_hash
P=Path('results/options_crowding_deleveraging_relay_support_v4_2026-08-08.json')
def test_support_artifact_is_terminal_before_novelty_or_economics():
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='1f84318cf2e822247b44aae780b4ac670479c07d20d7803ddb1003169fd7c60e'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==canonical_hash(core)=='aa1654e2f7ca80b42f4e64489de28658e6eda5c5ad161bc18cbbeea21ecda155'
 assert d['support_passed'] is False and d['decision']=='terminal_source_support_reject'
 assert d['gross9_novelty_status']=='not_authorized' and d['advance_to_economic_outcomes'] is False
 assert all(not d['support_checks'][f'{stage}_side_balance'] for stage in ('train','test','eval','final'))
