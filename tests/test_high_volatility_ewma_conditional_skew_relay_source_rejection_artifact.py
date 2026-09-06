import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_ewma_conditional_skew_relay_support_2026-08-11.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvewcs_source_rejection_is_terminal_and_outcomes_are_sealed():
 x=json.loads(R.read_text());assert x["support_passed"] is False and x["advance_to_gross9_novelty"] is False and x["advance_to_economic_outcomes"] is False and x["decision"]=="terminal_source_support_reject"
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":22,"test":41,"eval":40,"final":32}
 assert x["support_checks"]["train_side_balance"] is False and x["support_checks"]["eval_side_balance"] is False and x["support_checks"]["final_side_balance"] is False
 assert x["clock"]["rows"]==135 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="3c67b4c16644ea2dcab279d89a7d81d0baa424397da129b9d243de088128a5c4"
 assert sha(R)=="98ffc26e74e32983e25de32a22454c3681dfdb849b941265af89d884af245387"
 assert not Path("results/high_volatility_ewma_conditional_skew_relay_gross9_novelty_2026-08-11.json").exists()
