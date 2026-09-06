import hashlib,json
from training import build_deribit_led_shock_deceleration_reversal_support as s
def test_dlsdr_support_is_frozen_pass_before_novelty_and_economics():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=='62656838515078b69c8e6d4d95c0e2d7a55b64f8ea189dccb73b33a76ca64635'
 d=json.loads(s.RESULT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==s.chash(core)=='5700af902bb38ab765e89cf9e09e80157d20b45baf1913770ba3783ac7ec7fc2'
 assert d['clock']['sha256']==hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()
 assert d['support_passed'] is True and all(d['support_checks'].values())
 assert d['advance_to_gross9_novelty'] is True and d['advance_to_economic_outcomes'] is False
