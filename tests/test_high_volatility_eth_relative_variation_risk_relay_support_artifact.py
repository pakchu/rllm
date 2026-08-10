import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_eth_relative_variation_risk_relay_support_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_support_pass_frozen_blind():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False and x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False;assert {k:v["events"] for k,v in x["support"].items()}=={"train":85,"test":135,"eval":128,"final":72};assert x["clock"]["rows"]==420 and sha(Path(x["clock"]["path"]))=="3e4cad661098e0c82899eabf8d194cf05c3364302d3fd94e350ec533d62aa2db" and sha(R)=="53988f96e13b7c887e1b2c055ca07433598e9d990a5cab2bd0cf56a7265c020a"
