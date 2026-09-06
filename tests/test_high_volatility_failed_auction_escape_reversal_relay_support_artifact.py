import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_failed_auction_escape_reversal_relay_support_2026-08-11.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvfaer_support_pass_is_frozen_blind_reproducible():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":18,"test":22,"eval":32,"final":13} and all(x["support_checks"].values())
 assert x["clock"]["rows"]==85 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="5b7ba90030c340218ca10f70709eed65b1e19fe51d9451acc45b179c7a4cd394"
 assert sha(R)=="e82e5278ff9d830f948d3d579ad80bbf8c2bdab9cc4dfe0445b2861a199bbc61"
