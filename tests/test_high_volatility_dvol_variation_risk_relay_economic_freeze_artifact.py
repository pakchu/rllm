import json
from training import evaluate_high_volatility_dvol_variation_risk_relay_economics as economics
def test_economic_freeze_is_outcome_blind_and_binds_code():
 p=json.loads(economics.FREEZE.read_text());assert p["policy_id"]=="HVDVVR-12";assert not p["outcomes_opened"];assert p["evaluator"]["sha256"]==economics.sha256(economics.Path(economics.__file__));h=p.pop("manifest_hash");assert economics.canonical_hash(p)==h
