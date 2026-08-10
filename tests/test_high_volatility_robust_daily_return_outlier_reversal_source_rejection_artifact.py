import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_robust_daily_return_outlier_reversal_support_2026-08-11.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvrdor_source_rejection_is_terminal_and_outcomes_are_sealed():
 x=json.loads(R.read_text());assert x["support_passed"] is False and x["advance_to_gross9_novelty"] is False and x["advance_to_economic_outcomes"] is False and x["decision"]=="terminal_source_support_reject"
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":14,"test":43,"eval":34,"final":22}
 assert x["support_checks"]["train_month_concentration"] is False and sum(not v for v in x["support_checks"].values())==1
 assert x["clock"]["rows"]==113 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="b898a54ed7b6097f102dc9c2e50af64be719e37fdbc7e74fed0503c3caca42d5"
 assert sha(R)=="a9d23beae508311b338ea7754269a1690ab4e100531975c42b6fd8e3c16cf9d5"
 assert not Path("results/high_volatility_robust_daily_return_outlier_reversal_gross9_novelty_2026-08-11.json").exists()
