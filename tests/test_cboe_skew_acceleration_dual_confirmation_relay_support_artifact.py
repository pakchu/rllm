import hashlib,json
from training import build_cboe_skew_acceleration_dual_confirmation_relay_support as s
def test_cvskdmr_support_frozen_pass():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=="36ab77ad9cf6dc27553a92ad04720f454b0f986f16052415f0934dfe265fc6a8";assert hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()=="f912495583b14d30583738c4d6ecb728648532480fe548d0881725b5a030d9ed";p=json.loads(s.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==s.canonical_hash(core) and p["support_passed"] is True and p["advance_to_gross9_novelty"] is True;assert [p["support"][n]["events"] for n in s.SPLITS]==[31,60,72,52]
