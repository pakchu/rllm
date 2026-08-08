import hashlib,json
from training import build_cboe_vix_overnight_btc_confirmation_relay_support as s
def test_cvobr_support_frozen_pass():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=="f42e7d3447fdd65eba2f8fac986f66a9b3a1130013177b1d2d517733978d6540";assert hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()=="a23578f32046fc8247c1df3b42697eda1ea864c51d868efbb83bbc572414f98a";p=json.loads(s.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==s.canonical_hash(core) and p["support_passed"] is True and p["advance_to_gross9_novelty"] is True;assert p["advance_to_economic_outcomes"] is False and p["postentry_return_pnl_execution_price_opened"] is False;assert [p["support"][n]["events"] for n in s.SPLITS]==[13,37,33,12]
