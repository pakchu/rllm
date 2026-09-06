import hashlib,json
from training import build_cboe_crypto_vix_transmission_relay_support as support
def test_ccvtr_source_support_is_frozen_pass_without_outcomes():
 assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest()=="8e871969430deef3d027749df03836c0dc34b8e4b921e70755453cda4e184bfd"
 assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest()=="0f4dbeb084e10e91c684a4ec8f98b59980dd64d5115d7a2ae8e09f882c96e6ca"
 p=json.loads(support.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==support.canonical_hash(core)
 assert p["support_passed"] is True and p["advance_to_gross9_novelty"] is True and p["advance_to_economic_outcomes"] is False and p["decision"]=="pass_to_novelty"
 assert p["btc_price_postentry_return_pnl_opened"] is False and p["gross9_rows_opened"] is False
 assert [p["support"][n]["events"] for n in ("train","test","eval","final")]==[46,73,68,52]
 assert all(not x["promotion_authorized"] for x in p["controls"].values())
