import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_trade_count_dispersion_asymmetry_relay_support_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_support_pass_frozen_blind():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False and x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False;assert {k:v["events"] for k,v in x["support"].items()}=={"train":45,"test":53,"eval":67,"final":33};assert x["clock"]["rows"]==198 and sha(Path(x["clock"]["path"]))=="1bf7606a3b381e100b34154827c749be8a58649e9f2df26333d14f408ef66924" and sha(R)=="68a363ca7efb8b223a354ed26c9af867aab8ec83cfa53ea2457b4a868d116ef5"
