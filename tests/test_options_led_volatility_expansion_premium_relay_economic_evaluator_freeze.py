from __future__ import annotations
import hashlib,json
from pathlib import Path
from training import evaluate_options_led_volatility_expansion_premium_relay_economics as e
P=Path('results/options_led_volatility_expansion_premium_relay_economic_evaluator_freeze_2026-08-08.json')
def test_evaluator_freeze_is_hash_bound_and_outcome_blind()->None:
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='f94d86dc05e7db428bb79c38fe3902379ea78e2eb19e49889808c0d7a8be740d';d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==e.chash(core)=='f2e94d2ea925a374622078c7b33342204e7d98e9e80eaf3a28a0e3f942c90b27';assert d['evaluator']['sha256']=='3bda1c726e14bb511a5154e0f77972a114da804235ca863e098af83045502b87';assert d['outcomes_opened'] is False
