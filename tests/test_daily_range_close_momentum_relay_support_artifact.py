import hashlib,json
from training import build_daily_range_close_momentum_relay_support as support

def test_drcmr_source_support_is_frozen_pass_without_outcomes():
 assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest()=="31b6a2c74d3b7ec37513dc2e783ba3e9db5cdfbc0d56986de97c4f9a39fb925d";assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest()=="da215b7ed12695b5efa10d5b11b74dcd9f5162589b6984b7c82ed72030e19b50";p=json.loads(support.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==support.chash(core);assert p["support_passed"] is True and p["advance_to_gross9_novelty"] is True and p["advance_to_economic_outcomes"] is False and p["decision"]=="pass_to_novelty";assert p["postentry_return_pnl_execution_price_opened"] is False and p["gross9_rows_opened"] is False;assert [p["support"][s]["events"] for s in support.SPLITS]==[16,56,57,16];assert all(not x["promotion_authorized"] for x in p["controls"].values())
