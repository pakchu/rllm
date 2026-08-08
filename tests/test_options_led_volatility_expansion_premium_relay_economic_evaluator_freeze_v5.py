import hashlib,json
from pathlib import Path
from training import evaluate_options_led_volatility_expansion_premium_relay_economics_v5 as e

P=Path('results/options_led_volatility_expansion_premium_relay_economic_evaluator_freeze_v5_2026-08-08.json')

def test_v5_freeze_is_hash_bound_and_outcome_blind():
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='54a5b85b2ba68432d810c64faf41f2f3062ad0feaf026f4ec541e5cb04e2cec0'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==e.chash(core)=='426e625ccb46bb6e730a1c78350148bf2d3e0d4a2839edd01cff8bb47e5a5cd4'
 assert d['candidate_contract_change_from_v4'] is False and d['outcomes_opened'] is False
