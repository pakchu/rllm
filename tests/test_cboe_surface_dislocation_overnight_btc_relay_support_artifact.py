import hashlib,json
from training import build_cboe_surface_dislocation_overnight_btc_relay_support as s
def test_cvsdr_support_frozen_pass():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=="625e2ed5839fa77202513c264bd5205ee80bc0745b61ff749f8b9742acfb9350";assert hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()=="5858ed42548f6c65344d160001d5f97e3935c1d6bd97a8350b93180c95c03cdf";p=json.loads(s.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==s.canonical_hash(core) and p["support_passed"] is True and p["advance_to_gross9_novelty"] is True;assert p["advance_to_economic_outcomes"] is False;assert [p["support"][n]["events"] for n in s.SPLITS]==[52,88,97,54]
