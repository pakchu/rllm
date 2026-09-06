import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_wide_range_body_continuation_relay_support_2026-08-11.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvwrbc_support_pass_is_frozen_blind_reproducible():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":40,"test":109,"eval":86,"final":44} and all(x["support_checks"].values())
 assert x["clock"]["rows"]==279 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="66e69835212a7d9e0a0f330c9b3d2487157a69facc0301fad0e58e34e62986fc"
 assert sha(R)=="d5d093538bd489c2687ca5097eac67b7de4864137bed9715a0ed162d4e8b2745"
