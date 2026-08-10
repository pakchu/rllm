import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_opening_drive_hold_relay_support_2026-08-11.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvodhr_source_rejection_is_terminal_and_outcomes_are_sealed():
 x=json.loads(R.read_text());assert x["support_passed"] is False and x["advance_to_gross9_novelty"] is False and x["advance_to_economic_outcomes"] is False and x["decision"]=="terminal_source_support_reject"
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":24,"test":49,"eval":38,"final":21}
 assert x["support_checks"]["train_side_balance"] is False and sum(not v for v in x["support_checks"].values())==1
 assert x["clock"]["rows"]==132 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="f0b59084abe8f062cf11d9e0ceb4b1034c61360020a7f376d1cc4bdbf58af6d8"
 assert sha(R)=="64c84f4e06e4c697043cbef7ea59cd1923088f8b057a904dcd9d6b6e953e225c"
 assert not Path("results/high_volatility_opening_drive_hold_relay_gross9_novelty_2026-08-11.json").exists()
