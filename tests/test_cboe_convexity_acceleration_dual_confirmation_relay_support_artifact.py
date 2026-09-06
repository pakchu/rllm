import hashlib,json
from training import build_cboe_convexity_acceleration_dual_confirmation_relay_support as s
def test_cvdcmr_support_frozen_pass():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=="5b8a9c3f33d8c6419b8681ac4f87638efae91c31f2e321a5691441ec25df333f";assert hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()=="8fe7fcc0475b6fbaa35b8e398ba1a5ec6920bb41e3e2fbf3da0236f1f7242e60";p=json.loads(s.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==s.canonical_hash(core) and p["support_passed"] is True and p["advance_to_gross9_novelty"] is True;assert p["advance_to_economic_outcomes"] is False;assert [p["support"][n]["events"] for n in s.SPLITS]==[31,54,67,48]
