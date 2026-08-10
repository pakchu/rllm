import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_drawup_drawdown_asymmetry_relay_support_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvdudar_support_pass_is_frozen_blind_reproducible():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":64,"test":129,"eval":110,"final":58} and all(x["support_checks"].values())
 assert x["clock"]["rows"]==361 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="a46c75e13ae3a198ba93ffb0eddd62d640572a0523f464f05f1422800a02e725"
 assert sha(R)=="cd0d1e71b23baa8623edbcd4021b75855a74043230d3c04055f730d46260866b"
