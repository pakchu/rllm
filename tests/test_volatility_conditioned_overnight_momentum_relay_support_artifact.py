import hashlib,json
from training import build_volatility_conditioned_overnight_momentum_relay_support as s
def test_vomr_support_frozen_terminal():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=="3fee47064782021ed2740721f7b2162b409647e48590923a832071dcd9327f12";assert hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()=="b6504394b8af136b6f1a96608951a82d3756cc0893c5fbdb87e8d50d7c487d86";p=json.loads(s.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==s.canonical_hash(core) and p["support_passed"] is False and p["decision"]=="terminal_source_support_reject";assert p["advance_to_gross9_novelty"] is False;assert [p["support"][n]["events"] for n in s.SPLITS]==[4,35,16,7]
