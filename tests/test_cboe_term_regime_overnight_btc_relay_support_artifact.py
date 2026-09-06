import hashlib,json
from training import build_cboe_term_regime_overnight_btc_relay_support as s
def test_cvtbr_support_frozen_pass():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=="ce631b76f078837e7fb1278de876fae5055e3a93d61d33a2e86a4374651a82ce";assert hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()=="9ceb718fe3374e50d24e6df2b8e07f426217eb5779e6134ddc6254c66bddb32b";p=json.loads(s.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==s.canonical_hash(core) and p["support_passed"] is True and p["advance_to_gross9_novelty"] is True;assert p["advance_to_economic_outcomes"] is False and p["postentry_return_pnl_execution_price_opened"] is False;assert [p["support"][n]["events"] for n in s.SPLITS]==[47,97,106,49]
