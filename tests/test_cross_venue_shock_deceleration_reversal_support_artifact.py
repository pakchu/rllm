import hashlib,json
from training import build_cross_venue_shock_deceleration_reversal_support as s
def test_cvsdr_support_is_frozen_terminal_before_novelty_and_economics():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=='8e6d807b40754ece6dd8f628b51da2361de6b77caaedd86a1d6236bbe3d2c412'
 d=json.loads(s.RESULT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==s.chash(core)=='4a51188e7274ab18b00ea1d86df36088b7d1c32b46d9e75893ecf87deb233d53'
 assert d['support_checks']['eval_minimum_events'] is False and d['support_checks']['eval_month_concentration'] is False
 assert d['support_passed'] is False and d['advance_to_gross9_novelty'] is False and d['advance_to_economic_outcomes'] is False
