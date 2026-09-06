import hashlib,json
from training import evaluate_cboe_front_slope_crypto_volatility_transmission_relay_gross9_novelty as novelty
def test_cfstr_novelty_frozen_pass_without_outcomes():
 assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest()=="fc5d55f372d9fda5e267ff9d8ed3a89559a0f290ae62d94b69b3ac1297db7b9b";p=json.loads(novelty.OUTPUT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==novelty.chash(core) and p["every_gross9_sleeve_passed"] is True and p["advance_to_economic_outcomes"] is True and p["evidence_boundary"]["outcomes_opened"] is False
