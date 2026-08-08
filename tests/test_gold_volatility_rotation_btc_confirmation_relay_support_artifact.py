import hashlib,json
from training import build_gold_volatility_rotation_btc_confirmation_relay_support as s
def test_gvrcr_support_is_frozen_pass_before_novelty_and_economics():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=='4811e41caff144bb7738d36c6d579d04d03d22913ae869615ac198251ccf7fe1';d=json.loads(s.RESULT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==s.canonical_hash(core)=='6b832fd695b0b5bc4eb7848889cc27b001b48258456d1580857028b159206e49';assert d['clock']['rows']==181 and d['support_passed'] is True and all(d['support_checks'].values()) and d['advance_to_gross9_novelty'] is True and d['advance_to_economic_outcomes'] is False
