import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_engulfing_auction_acceptance_relay_support_2026-08-11.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hveaar_source_rejection_is_terminal_and_outcomes_are_sealed():
 x=json.loads(R.read_text());assert x["support_passed"] is False and x["advance_to_gross9_novelty"] is False and x["advance_to_economic_outcomes"] is False and x["decision"]=="terminal_source_support_reject"
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":7,"test":14,"eval":12,"final":13}
 assert x["support_checks"]["train_minimum_events"] is False and sum(not v for v in x["support_checks"].values())==1
 assert x["clock"]["rows"]==46 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="22ddbacb766a7bc129ea1620896cfd7067673345900be5237f6e391019429510"
 assert sha(R)=="cd59828a45d768895ecfc627aef7bbaa31ffb121d5aa5f303eb2cab3252cf97f"
 assert not Path("results/high_volatility_engulfing_auction_acceptance_relay_gross9_novelty_2026-08-11.json").exists()
