import hashlib,json
from pathlib import Path
F=Path("results/high_volatility_quote_turnover_concentration_continuation_relay_economic_evaluator_freeze_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ch(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def test_hvtccr_economic_freeze_is_outcome_blind_and_code_bound():
 x=json.loads(F.read_text());core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==ch(core);assert x["outcomes_opened"] is False;assert x["empty_diagnostic_controls_handled_before_outcomes"] is True
 assert sha(Path(x["evaluator"]["path"]))==x["evaluator"]["sha256"];assert sha(Path(x["authorization"]["path"]))==x["authorization"]["sha256"]
