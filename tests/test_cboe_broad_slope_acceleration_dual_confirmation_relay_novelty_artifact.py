import hashlib,json
from training import evaluate_cboe_broad_slope_acceleration_dual_confirmation_relay_gross9_novelty as n
def test_cvbdmr_novelty_frozen_pass():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=="7742eaf52053ce5a9a7bfcd9b30063a6f58dc38dd7c5a1be88f1234a1b5f2400";p=json.loads(n.OUTPUT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==n.chash(core) and p["every_gross9_sleeve_passed"] is True and p["advance_to_economic_outcomes"] is True;assert max(x["metrics"]["one_to_one_6h_max_matched_share"] for x in p["gross9_sleeves"].values())<.18
