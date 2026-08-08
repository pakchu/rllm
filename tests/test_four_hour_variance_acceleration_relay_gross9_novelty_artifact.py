import hashlib,json
from training import evaluate_four_hour_variance_acceleration_relay_gross9_novelty as novelty
def test_fhvar_novelty_artifact_passes_without_outcomes():
 assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest()=="811bb91d7ba2b5472f55bdb3e481a88f212da3f94a06254d784cfc8c0f1a22ad";p=json.loads(novelty.OUTPUT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==novelty.chash(core);assert p["gross9_novelty_status"]=="passed" and p["every_gross9_sleeve_passed"] is True and p["advance_to_economic_outcomes"] is True;assert p["evidence_boundary"]["outcomes_opened"] is False and p["evidence_boundary"]["btc_price_or_return_rows_opened"]==0
