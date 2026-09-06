import hashlib,json
from training import evaluate_cross_venue_relative_volatility_premium_regime_crossing_relay_gross9_novelty as n
def test_cvrvpr_novelty_is_frozen_pass_before_economics():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=='4432ee2922773d33866421edc12d76e724536060a81e4493b6c8f7e9e68d7466'
 d=json.loads(n.OUTPUT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==n.canonical_hash(core)=='3e8bb1d55f5f181f05e07b3e9e99f7ee2a29f4070c46b94ce174acd3456b503a'
 assert d['every_gross9_sleeve_passed'] is True and d['advance_to_economic_outcomes'] is True and d['evidence_boundary']['outcomes_opened'] is False
