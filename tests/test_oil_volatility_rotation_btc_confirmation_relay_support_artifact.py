import hashlib,json
from training import build_oil_volatility_rotation_btc_confirmation_relay_support as s
def test_ovrcr_support_is_frozen_pass_before_novelty_and_economics():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=='e8ca7c0136f5126f29a1412c485c0750031d36392e1912e663dc7949945fb68e';d=json.loads(s.RESULT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==s.canonical_hash(core)=='794e50c1f57e1beb30960a8f30da580abbbf4bba739f9f8e3a30f6781e3340e9';assert d['clock']['rows']==166 and d['support_passed'] is True and all(d['support_checks'].values()) and d['advance_to_gross9_novelty'] is True and d['advance_to_economic_outcomes'] is False
