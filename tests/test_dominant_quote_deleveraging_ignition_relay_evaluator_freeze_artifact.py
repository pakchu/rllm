import hashlib,json
from pathlib import Path
from training import evaluate_dominant_quote_deleveraging_ignition_relay_economics as economics
def test_dqdir_outcome_blind_evaluator_freeze_is_bound():
 assert hashlib.sha256(economics.FREEZE.read_bytes()).hexdigest()=="29a06e42af4820cbc53608537a312560c65105081f0116009d7ac91f4423c957"
 d=json.loads(economics.FREEZE.read_text());core={k:v for k,v in d.items() if k!="manifest_hash"};assert d["manifest_hash"]==economics.canonical_hash(core)
 assert d["outcomes_opened"] is False and d["evaluator"]["sha256"]==economics.sha256(Path(economics.__file__))
 novelty,freeze=economics.verify("train");assert novelty["advance_to_economic_outcomes"] is True and freeze["outcomes_opened"] is False
