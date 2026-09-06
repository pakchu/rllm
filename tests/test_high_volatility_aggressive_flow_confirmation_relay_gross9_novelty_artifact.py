import hashlib,json
from training import evaluate_high_volatility_aggressive_flow_confirmation_relay_gross9_novelty as novelty
def test_hvafc_novelty_artifact_passes_without_outcomes():
 assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest()=="39ae6ab9da84cd37e42fbb792b3e8b6bb13048c2e9244abb1fec7c60679b128b";p=json.loads(novelty.OUTPUT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==novelty.chash(core);assert p["gross9_novelty_status"]=="passed" and p["every_gross9_sleeve_passed"] is True and p["advance_to_economic_outcomes"] is True;assert p["evidence_boundary"]["outcomes_opened"] is False and p["evidence_boundary"]["btc_price_or_return_rows_opened"]==0
