import hashlib,json
from training import build_cboe_vix_acceleration_dual_confirmation_relay_support as s
def test_cvvdmr_support_frozen_pass():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=="30267913937b9f2032eabdaaf401c7b640ef133ad5e56d1a1e7163c44bd57e14";assert hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()=="624a693cc972a168339fb8251239ab1a2ef78ce88cf0e1040f4340e509bad27d";p=json.loads(s.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==s.canonical_hash(core) and p["support_passed"] is True and p["advance_to_gross9_novelty"] is True;assert [p["support"][n]["events"] for n in s.SPLITS]==[24,58,57,38]
