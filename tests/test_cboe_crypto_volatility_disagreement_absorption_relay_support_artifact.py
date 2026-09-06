import hashlib,json
from training import build_cboe_crypto_volatility_disagreement_absorption_relay_support as support
def test_ccvdar_source_support_is_frozen_pass_without_outcomes():
 assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest()=="142203c4cb408860564ac3aa70a939b654348512a9938654b2f0bd88c23a71e0"
 assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest()=="14ff7b1406be9a147eebda2746cf2c2a71f60c23cdb90928e6189cfb2bb19ee3"
 p=json.loads(support.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==support.canonical_hash(core)
 assert p["support_passed"] is True and p["advance_to_gross9_novelty"] is True and p["advance_to_economic_outcomes"] is False and p["decision"]=="pass_to_novelty"
 assert p["btc_price_postentry_return_pnl_opened"] is False and p["gross9_rows_opened"] is False
 assert [p["support"][n]["events"] for n in ("train","test","eval","final")]==[49,76,76,61]
 assert all(not x["promotion_authorized"] for x in p["controls"].values())
