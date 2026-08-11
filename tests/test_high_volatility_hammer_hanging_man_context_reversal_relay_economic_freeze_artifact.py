import json
from pathlib import Path
from training import evaluate_high_volatility_hammer_hanging_man_context_reversal_relay_economics as e
def test_freeze_is_blind_and_bound():
 v=json.loads(e.FREEZE.read_text());core={k:x for k,x in v.items() if k!="manifest_hash"};assert v["manifest_hash"]==e.canonical_hash(core) and v["outcomes_opened"] is False and v["empty_diagnostic_controls_handled_before_outcomes"] is True
 assert e.sha256(Path(v["evaluator"]["path"]))==v["evaluator"]["sha256"] and e.sha256(Path(v["authorization"]["path"]))==v["authorization"]["sha256"]
