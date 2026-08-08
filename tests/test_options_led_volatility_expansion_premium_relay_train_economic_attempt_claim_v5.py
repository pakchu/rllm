import hashlib,json
from pathlib import Path
from training import evaluate_options_led_volatility_expansion_premium_relay_economics_v5 as e

P=Path('results/options_led_volatility_expansion_premium_relay_train_economic_attempt_claim_v5_2026-08-08.json')

def test_v5_claim_is_hash_bound_and_later_windows_are_sealed():
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='3ed8ed58f33d3b7d24461015ccd7ab042be798d50ea77321072279983bdbd435'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==e.chash(core)=='ad2277772e08fb15d595d3ec145ac5329713f66c5f8b5a8a1e4e951e16870fe3'
 assert d['outcomes_opened'] is False and len(d['sealed_later_windows'])==3
