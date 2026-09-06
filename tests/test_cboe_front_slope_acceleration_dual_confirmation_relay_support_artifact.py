import hashlib,json
from training import build_cboe_front_slope_acceleration_dual_confirmation_relay_support as s
def test_cvfdmr_support_frozen_pass():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=="d43f819b78c8b5009b46b4bc9599197b8390c6a725eea4a03b873e6da80083b2";assert hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()=="04e077239a23e70966d94b13ad7fa8a81f41fe8f3cc881b1728432dce4b6cc24";p=json.loads(s.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==s.canonical_hash(core) and p["support_passed"] is True and p["advance_to_gross9_novelty"] is True;assert [p["support"][n]["events"] for n in s.SPLITS]==[25,58,60,41]
