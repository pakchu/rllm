import hashlib,json
from training import build_cross_venue_relative_volatility_premium_regime_crossing_relay_support as s
def test_cvrvpr_support_is_frozen_pass_before_novelty_and_economics():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=='a8fec64c957be3680ac783b6481fa7e727c8f90d48a0b50fe355bcaa3eb0d924'
 d=json.loads(s.RESULT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==s.chash(core)=='4c5ceb135cb76ad1b9f24ddbc04504910b1869a1682feb574bf0abc3adff8e1a'
 assert d['clock']['sha256']==hashlib.sha256(s.CLOCK.read_bytes()).hexdigest() and d['clock']['rows']==416
 assert d['support_passed'] is True and all(d['support_checks'].values()) and d['advance_to_gross9_novelty'] is True and d['advance_to_economic_outcomes'] is False
