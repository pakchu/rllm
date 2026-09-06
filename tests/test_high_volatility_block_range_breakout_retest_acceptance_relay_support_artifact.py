import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_block_range_breakout_retest_acceptance_relay_support_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvrbrar_support_pass_is_frozen_blind_reproducible():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":80,"test":157,"eval":167,"final":78} and all(x["support_checks"].values())
 assert x["clock"]["rows"]==482 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="77b16b08f928883c67a408304f5fbb694ced3c378534986efe9020f90cc1fc98"
 assert sha(R)=="64e7abc1b7283cbe3b7e4dad010594f028b021e711ff25ba65eba57a8c2fad67"
