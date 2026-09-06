import hashlib,json
from training import evaluate_nasdaq_volatility_rotation_btc_confirmation_relay_gross9_novelty as n
def test_nvxcr_novelty_is_frozen_pass_before_economics():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=='36a8c1a7d1c641b03ee3b8d1da32d73078cfb3d5a98d3e0a6805670bfdcc79ca';d=json.loads(n.OUTPUT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==n.canonical_hash(core)=='b451c3e002024507bc1c9da235e3bb829e76e6bd003877211799f6a290cdee83';assert d['every_gross9_sleeve_passed'] is True and d['advance_to_economic_outcomes'] is True and d['evidence_boundary']['outcomes_opened'] is False
