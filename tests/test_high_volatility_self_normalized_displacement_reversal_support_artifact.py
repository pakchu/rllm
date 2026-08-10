import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_self_normalized_displacement_reversal_support_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_support_pass_frozen_blind():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False and x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False;assert {k:v["events"] for k,v in x["support"].items()}=={"train":57,"test":110,"eval":86,"final":47};assert x["clock"]["rows"]==300 and sha(Path(x["clock"]["path"]))=="ba51e11fc949e9c04fa344ea9f882648fa0619735393a3f4e61189d6d4a8720d" and sha(R)=="77fdc7feee648b75b3f6aea58bf2eecfcf3a7933ecef5caea5e0d5e5dafb85aa"
