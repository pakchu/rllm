import hashlib,json
from pathlib import Path
from training import evaluate_cboe_crypto_volatility_disagreement_absorption_relay_economics as economics
def test_ccvdar_outcome_blind_evaluator_freeze_is_bound():
 assert hashlib.sha256(economics.FREEZE.read_bytes()).hexdigest()=="453ca18faab4549110d948709759c6442690bd1cb2e4db2bc24ba570829097ec"
 d=json.loads(economics.FREEZE.read_text());core={k:v for k,v in d.items() if k!="manifest_hash"};assert d["manifest_hash"]==economics.canonical_hash(core)
 assert d["outcomes_opened"] is False and d["evaluator"]["sha256"]==economics.sha256(Path(economics.__file__))
 novelty,freeze=economics.verify("train");assert novelty["advance_to_economic_outcomes"] is True and freeze["outcomes_opened"] is False
