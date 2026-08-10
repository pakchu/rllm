import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_cross_alt_wick_consensus_relay_support_2026-08-10.json")
def test_terminal_source_rejection():
 d=json.loads(RESULT.read_text());assert d["support_passed"] is False and d["decision"]=="terminal_source_support_reject" and d["advance_to_gross9_novelty"] is False and d["advance_to_economic_outcomes"] is False
 assert {k:v["events"] for k,v in d["support"].items()}=={"train":25,"test":51,"eval":54,"final":27}
 assert d["support"]["train"]["minority_side_share"]==.08 and d["support_checks"]["train_side_balance"] is False and d["support_checks"]["test_side_balance"] is False
 assert d["postentry_return_pnl_execution_price_opened"] is False and d["funding_values_opened"] is False and d["gross9_rows_opened"] is False
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()=="da7c3c12d02ea53f598530666ac44be1cf17464318a0af8a335e00b5ed95b07c"
def test_downstream_sealed():
 assert not Path("results/high_volatility_cross_alt_wick_consensus_relay_gross9_novelty_2026-08-10.json").exists()
 for stage in ("train","test","eval","final"):assert not Path(f"results/high_volatility_cross_alt_wick_consensus_relay_{stage}_economics_2026-08-10.json").exists()
