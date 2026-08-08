import json
from training import build_spot_led_volatility_catchup_relay_support as support
def test_slvcr_support_is_terminal_before_novelty_and_outcomes():
 p=json.loads(support.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==support.chash(core);assert p["support_passed"] is False and p["advance_to_gross9_novelty"] is False and p["advance_to_economic_outcomes"] is False;assert p["postentry_return_pnl_execution_price_opened"] is False and p["gross9_rows_opened"] is False;assert [p["support"][n]["events"] for n in ("train","test","eval","final")]==[0,0,0,2];assert p["decision"]=="terminal_source_support_reject"
