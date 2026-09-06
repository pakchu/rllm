import hashlib,json
from training import build_cboe_volatility_hour_momentum_relay_support as s
def test_cvhmr_support_frozen_terminal():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=="f411397921eab84dc4bd02ae885f8c3f397095929436bd524798da637a8f5d10";assert hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()=="3fa3bdb61d4d6edd779fad495f16a1b3f5de1a42be534001b33582a476afd43b";p=json.loads(s.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==s.canonical_hash(core) and p["support_passed"] is False and p["decision"]=="terminal_source_support_reject";assert p["advance_to_gross9_novelty"] is False;assert [p["support"][n]["events"] for n in s.SPLITS]==[49,66,48,14];assert p["support"]["final"]["max_month_share"]>.70
