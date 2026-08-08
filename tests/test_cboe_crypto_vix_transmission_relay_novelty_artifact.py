import hashlib,json
from training import evaluate_cboe_crypto_vix_transmission_relay_gross9_novelty as novelty
def test_ccvtr_novelty_frozen_pass_without_outcomes():
 assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest()=="5eff04b87752230b1da27f55d835eac01487f4075717c014dc1f842afd0c7d88"
 p=json.loads(novelty.OUTPUT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==novelty.chash(core)
 assert p["every_gross9_sleeve_passed"] is True and p["advance_to_economic_outcomes"] is True and p["evidence_boundary"]["outcomes_opened"] is False
 assert max(x["metrics"]["one_to_one_6h_max_matched_share"] for x in p["gross9_sleeves"].values())<=.16
