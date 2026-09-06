import hashlib,json
from training import evaluate_cboe_front_slope_acceleration_dual_confirmation_relay_gross9_novelty as n
def test_cvfdmr_novelty_frozen_pass():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=="9e96149733715789e327a8e4a8271b381641ce379e947f2705a875e5daa26b67";p=json.loads(n.OUTPUT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==n.chash(core) and p["every_gross9_sleeve_passed"] is True and p["advance_to_economic_outcomes"] is True;assert p["evidence_boundary"]["outcomes_opened"] is False;assert max(x["metrics"]["one_to_one_6h_max_matched_share"] for x in p["gross9_sleeves"].values())<.12
