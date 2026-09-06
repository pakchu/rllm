import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_multifractal_coherence_relay_support_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_support_pass_is_frozen_blind_reproducible():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False and x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":43,"test":78,"eval":77,"final":40};assert x["clock"]["rows"]==238 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="2c46dc9ec3974c2126f80ebdcff46cbe2f0d8f63db510acefd6ba8466fb106f0";assert sha(R)=="e0a6938cee24f306f364dea27d0ce066f0b54cb029458683cc8e1eb48358fe23"
