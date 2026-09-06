import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_flat_auction_absorption_relay_support_2026-08-10.json")
def test_terminal_source_rejection():
 d=json.loads(RESULT.read_text());assert d["support_passed"] is False and d["decision"]=="terminal_source_support_reject" and d["advance_to_gross9_novelty"] is False and d["advance_to_economic_outcomes"] is False
 assert d["postentry_return_pnl_execution_price_opened"] is False and d["funding_values_opened"] is False and d["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in d["support"].items()}=={"train":1,"test":3,"eval":2,"final":9}
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()=="bafb7465a695353e1e0ee3e45741e7dc93d6ed57d64c87c6927ef427933965f3"
def test_downstream_sealed():
 assert not Path("results/high_volatility_flat_auction_absorption_relay_gross9_novelty_2026-08-10.json").exists()
 for stage in ("train","test","eval","final"):assert not Path(f"results/high_volatility_flat_auction_absorption_relay_{stage}_economics_2026-08-10.json").exists()
