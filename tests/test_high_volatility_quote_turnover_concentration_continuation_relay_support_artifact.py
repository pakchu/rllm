import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_quote_turnover_concentration_continuation_relay_support_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvtccr_support_pass_is_frozen_blind_and_reproducible():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":41,"test":74,"eval":85,"final":33}
 assert x["clock"]["rows"]==233 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]
 assert sha(R)=="d874a611ae2ea65637fbbf7ca607062cc3df9c4d5989019bb23bf1abbb2713b3"
