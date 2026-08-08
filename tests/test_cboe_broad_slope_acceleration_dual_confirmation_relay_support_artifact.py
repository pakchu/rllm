import hashlib,json
from training import build_cboe_broad_slope_acceleration_dual_confirmation_relay_support as s
def test_cvbdmr_support_frozen_pass():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=="018246e128aac3314312ab100fea5740d091d27c018166879e65ef34e36450d7";assert hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()=="d6e658de2733aef6e90ff9935514e649423f846b0bddcfdf8693c4a2c73c1771";p=json.loads(s.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==s.canonical_hash(core) and p["support_passed"] is True and p["advance_to_gross9_novelty"] is True;assert [p["support"][n]["events"] for n in s.SPLITS]==[29,65,58,39]
