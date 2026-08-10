import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_quartile_median_staircase_relay_support_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvqmsr_support_pass_is_frozen_blind_reproducible():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":55,"test":89,"eval":77,"final":48} and all(x["support_checks"].values())
 assert x["clock"]["rows"]==269 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="c868691c01011cee7293e3d8cf3ddeb3aa5a9fa2ba18984a210038a7d0912161"
 assert sha(R)=="8d120b95b2a41ecd228994a25ff762b5d946c4f0c2e3f7cbcee8862a28fe598c"
