import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_cross_alt_range_expansion_confirmation_relay_economics as e
F=Path("results/high_volatility_cross_alt_range_expansion_confirmation_relay_economic_evaluator_freeze_2026-08-11.json")
def test_freeze_is_outcome_blind_and_bound():
 x=json.loads(F.read_text());core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==e.canonical_hash(core)
 assert x["outcomes_opened"] is False and x["evaluator"]["sha256"]==hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest()
 assert x["authorization"]["sha256"]==e.NOVELTY_SHA and x["empty_diagnostic_controls_handled_before_outcomes"] is True
