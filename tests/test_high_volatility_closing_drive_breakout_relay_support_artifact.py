import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_closing_drive_breakout_relay_support_2026-08-11.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvcdbr_support_pass_is_frozen_blind_reproducible():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":27,"test":60,"eval":63,"final":40} and all(x["support_checks"].values())
 assert x["clock"]["rows"]==190 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="b9a90a696b55d14ae1ce90fb0c77429a5bc8a95f13aaf08cbcb6b0608752f722"
 assert sha(R)=="5d8413ebf4a81c2fd79e05723643b43236091317fbd675deba66b16fab5aea23"
