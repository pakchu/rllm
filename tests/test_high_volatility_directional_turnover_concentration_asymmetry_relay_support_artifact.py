import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_directional_turnover_concentration_asymmetry_relay_support_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvdtcar_support_pass_is_frozen_blind_reproducible():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":62,"test":95,"eval":99,"final":43}
 assert all(x["support_checks"].values()) and x["clock"]["rows"]==299
 assert sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="daeb113cd6a0c316545c5e888eb355fdf03bfd88f03c079737df047479031d1a"
 assert sha(R)=="0649e9812f4382bb56bdcf13e8355f88f9e022cef18ca2602ef1448bbcb7e0ac"
