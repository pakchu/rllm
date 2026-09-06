import json
from pathlib import Path
from training import evaluate_cboe_convexity_crypto_volatility_transmission_relay_economics as economics
def test_ccxtr_evaluator_frozen_before_outcomes():
 p=json.loads(economics.FREEZE.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==economics.canonical_hash(core) and p["outcomes_opened"] is False and p["stop_on_first_failure"] is True and p["evaluator"]["sha256"]==economics.sha256(Path(economics.__file__))
