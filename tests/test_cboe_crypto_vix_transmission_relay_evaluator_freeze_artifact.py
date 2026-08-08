import hashlib,json
from pathlib import Path
from training import evaluate_cboe_crypto_vix_transmission_relay_economics as economics
def test_ccvtr_outcome_blind_evaluator_freeze_is_bound():
 assert hashlib.sha256(economics.FREEZE.read_bytes()).hexdigest()=="882a20c91b2ea8f8a9ee0586bff30d29aea75a94da636170253e7c9bc05a6cad"
 d=json.loads(economics.FREEZE.read_text());core={k:v for k,v in d.items() if k!="manifest_hash"};assert d["manifest_hash"]==economics.canonical_hash(core) and d["outcomes_opened"] is False and d["evaluator"]["sha256"]==economics.sha256(Path(economics.__file__))
 novelty,freeze=economics.verify("train");assert novelty["advance_to_economic_outcomes"] is True and freeze["outcomes_opened"] is False
