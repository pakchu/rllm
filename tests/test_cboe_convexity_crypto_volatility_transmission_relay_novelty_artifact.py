import hashlib,json
from training import evaluate_cboe_convexity_crypto_volatility_transmission_relay_gross9_novelty as novelty
def test_ccxtr_novelty_frozen_pass_without_outcomes():
 assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest()=="a9a3aa268c074342b8447a6df4d0576e3ad1005ad529c14c56fd132c7901f0b4";p=json.loads(novelty.OUTPUT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==novelty.chash(core);assert p["every_gross9_sleeve_passed"] is True and p["advance_to_economic_outcomes"] is True and p["evidence_boundary"]["outcomes_opened"] is False
