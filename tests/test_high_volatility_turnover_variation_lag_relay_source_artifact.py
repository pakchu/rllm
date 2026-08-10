import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_turnover_variation_lag_relay_support_2026-08-10.json")
def test_terminal_source_rejection():
 d=json.loads(RESULT.read_text());assert d["support_passed"] is False and d["decision"]=="terminal_source_support_reject" and d["advance_to_gross9_novelty"] is False and d["advance_to_economic_outcomes"] is False
 assert {k:v["events"] for k,v in d["support"].items()}=={"train":10,"test":26,"eval":17,"final":16}
 assert d["support"]["train"]["max_month_share"]==.5 and d["support_checks"]["train_month_concentration"] is False
 assert d["postentry_return_pnl_execution_price_opened"] is False and d["funding_values_opened"] is False and d["gross9_rows_opened"] is False
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()=="b6bc744bb947ba0a1af3016bb23b4ab59a182c548fb41263fb5016a2c8488dd7"
def test_downstream_sealed():
 assert not Path("results/high_volatility_turnover_variation_lag_relay_gross9_novelty_2026-08-10.json").exists()
 for stage in ("train","test","eval","final"):assert not Path(f"results/high_volatility_turnover_variation_lag_relay_{stage}_economics_2026-08-10.json").exists()
