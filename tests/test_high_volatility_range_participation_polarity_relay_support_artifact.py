import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_range_participation_polarity_relay_support_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_support_pass_is_frozen_blind_reproducible():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False and x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":64,"test":97,"eval":87,"final":38};assert x["clock"]["rows"]==286 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="7fa67b1ddaeb7b2e02840f9f96011333763f78ddffd7ce47a2bccf4f05c9e21d";assert sha(R)=="b506b3201380e7b3db5e62406371d566910e47d639b7b0168265457eba379ed4"
