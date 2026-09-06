import json
from training import evaluate_high_volatility_epu_hedge_relay_economics as economics
def test_freeze_is_outcome_blind_and_binds_evaluator():
 p=json.loads(economics.FREEZE.read_text());assert p["policy_id"]=="HVEPUH-24";assert not p["outcomes_opened"];assert p["evaluator"]["sha256"]==economics.sha256(economics.Path(economics.__file__));h=p.pop("manifest_hash");assert economics.canonical_hash(p)==h
