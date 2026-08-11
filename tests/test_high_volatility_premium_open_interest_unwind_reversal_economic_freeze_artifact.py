import hashlib, json
from pathlib import Path
from training import evaluate_high_volatility_premium_open_interest_unwind_reversal_economics as e


def test_evaluator_was_frozen_outcome_blind():
    value=json.loads(e.FREEZE.read_text()); core={k:v for k,v in value.items() if k!='manifest_hash'}
    assert value['manifest_hash']==e.canonical_hash(core)
    assert value['outcomes_opened'] is False
    assert value['empty_clock_policy'].startswith('outcome-blind load_clock_allow_empty')
    assert value['evaluator']['sha256']==hashlib.sha256(Path(value['evaluator']['path']).read_bytes()).hexdigest()
    assert value['authorization']['sha256']==e.NOVELTY_SHA
