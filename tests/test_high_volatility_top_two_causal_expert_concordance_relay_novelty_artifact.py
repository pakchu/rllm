import hashlib,json
from training import evaluate_high_volatility_top_two_causal_expert_concordance_relay_gross9_novelty as n

def test_hvtcec_novelty_is_frozen_pass_before_economics():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=='591a88d2730ca6864cc29a28ba562ddbf1a6b5e79ff9dccf8b06800cb0a2d1a0'
 result=json.loads(n.OUTPUT.read_text());core={k:v for k,v in result.items() if k!='manifest_hash'}
 assert result['manifest_hash']==n.canonical_hash(core)=='bd5c291b49d5892e06cec486c22e1129c99e4fed67dd477ed58ba706579061ac'
 assert result['every_gross9_sleeve_passed'] is True and result['advance_to_economic_outcomes'] is True
 assert result['evidence_boundary']['outcomes_opened'] is False
