import hashlib,json
from training import build_volatility_gated_stablecoin_sequential_follower_relay_support as support

def test_vgsfr_source_support_is_frozen_terminal():
 assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest()=="0166bd1256ed2c8d13b1ff7f592f886dc3b81ea1cef2a65fa75f1b1afeb28df1"
 assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest()=="5ede00b9cd544fb8b71aa5a20724c521f288d7354f45394af24292428ddd6bbc"
 payload=json.loads(support.RESULT.read_text());core={k:v for k,v in payload.items() if k!="manifest_hash"}
 assert payload["manifest_hash"]==support.chash(core)
 assert payload["support_passed"] is False and payload["advance_to_gross9_novelty"] is False
 assert payload["advance_to_economic_outcomes"] is False and payload["decision"]=="terminal_source_support_reject"
 assert payload["postentry_return_pnl_execution_price_opened"] is False and payload["gross9_rows_opened"] is False
 assert [payload["support"][s]["events"] for s in ("train","test","eval","final")]==[1,2,0,2]
