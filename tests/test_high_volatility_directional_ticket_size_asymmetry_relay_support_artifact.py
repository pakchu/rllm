import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_directional_ticket_size_asymmetry_relay_support_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_support_pass_frozen_blind():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False and x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False;assert {k:v["events"] for k,v in x["support"].items()}=={"train":32,"test":67,"eval":70,"final":47};assert x["clock"]["rows"]==216 and sha(Path(x["clock"]["path"]))=="f4ec92ddd8b3b6ff78554a46098c861986422b3f07e7d24cb7aed7b256657020" and sha(R)=="8e5a956ebf54056f3363afb5b484566b4aa96979a5dcdb8522bf900c2cb917f1"
