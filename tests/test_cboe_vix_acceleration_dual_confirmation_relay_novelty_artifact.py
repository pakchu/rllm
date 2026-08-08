import hashlib,json
from training import evaluate_cboe_vix_acceleration_dual_confirmation_relay_gross9_novelty as n
def test_cvvdmr_novelty_frozen_pass():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=="bae3acc00b0b51ac8369b3cdafe2576e0249ae6ba56acb8fc27aa2206146f9b2";p=json.loads(n.OUTPUT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==n.chash(core) and p["every_gross9_sleeve_passed"] is True and p["advance_to_economic_outcomes"] is True;assert max(x["metrics"]["one_to_one_6h_max_matched_share"] for x in p["gross9_sleeves"].values())<.18
