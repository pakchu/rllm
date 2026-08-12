import hashlib, json
from pathlib import Path
from training import evaluate_confirmation_ladder_settlement_dispersion_escalation_relay_economics as economics


def test_freeze_is_outcome_blind_bound_and_empty_safe():
    freeze=json.loads(economics.FREEZE.read_text()); core={k:v for k,v in freeze.items() if k!="manifest_hash"}
    digest=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    assert freeze["manifest_hash"] == digest
    assert freeze["outcomes_opened"] is False
    assert freeze["load_clock_allow_empty"] is True
    assert freeze["evaluator"]["sha256"] == economics.sha256(Path(economics.__file__))
    assert freeze["authorization"]["sha256"] == economics.NOVELTY_SHA
