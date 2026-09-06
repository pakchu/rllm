import json
from training import build_spot_participation_volatility_ignition_relay_support as support
def test_spvir_support_pass_is_hash_bound_and_outcome_blind():
 p=json.loads(support.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==support.chash(core);assert p["support_passed"] is True and p["advance_to_gross9_novelty"] is True and p["advance_to_economic_outcomes"] is False;assert p["postentry_return_pnl_execution_price_opened"] is False and p["gross9_rows_opened"] is False and all(p["support_checks"].values());assert [p["support"][n]["events"] for n in ("train","test","eval","final")]==[222,397,407,216]
