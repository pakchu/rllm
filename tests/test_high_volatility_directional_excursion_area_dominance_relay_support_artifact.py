import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_directional_excursion_area_dominance_relay_support_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvdeadr_support_pass_is_frozen_blind_reproducible():
 x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert {k:v["events"] for k,v in x["support"].items()}=={"train":49,"test":85,"eval":81,"final":47} and all(x["support_checks"].values())
 assert x["clock"]["rows"]==262 and sha(Path(x["clock"]["path"]))==x["clock"]["sha256"]=="eefaced770338127c1274f5dfcc9c6bb8b4475e841ff97bb1dda1f98dec4e857"
 assert sha(R)=="6dd3d5567eea0057b9d31627ccff0e11813b9bb53ac153f98d1564f45c896851"
