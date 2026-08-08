import hashlib,json
from training import evaluate_realized_over_implied_volatility_shock_continuation_relay_gross9_novelty as n
def test_rivscr_novelty_is_frozen_pass_before_economics():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=='78ec7720c638b323f3ce1e50a07cbcb9d9dd3b7dcdcfd52c261460c5c3ca55c6';d=json.loads(n.OUTPUT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==n.canonical_hash(core)=='cae43a72b61bd43cd4203b4bc644c3a75c20cbf34716ceab8f9ef83041954ef7';assert d['every_gross9_sleeve_passed'] is True and d['advance_to_economic_outcomes'] is True and d['evidence_boundary']['outcomes_opened'] is False
