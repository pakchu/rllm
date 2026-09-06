import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_inside_auction_resolution_relay_support_2026-08-11.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hviarr_support_pass_is_frozen_blind_reproducible():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":9,"test":26,"eval":18,"final":13} and all(x["support_checks"].values())
 assert x["clock"]["rows"]==66 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="2a1f3514825d29666bbdbd28d499ba6b8faa7211ecf154e7198b7a0fe467d4e5"
 assert sha(R)=="4ba725bd220d42fee3d68ebb6a7b0dc7d4df7d19ecb88f243a048cf3cfeb1b0f"
