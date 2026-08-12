import hashlib,json
from training import evaluate_high_volatility_leverage_displacement_amplification_reversal_gross9_novelty as n
EXPECTED='00371cd034cfe7bfa968ce328779f240b2d4954383646cc0e5ca1da0190d3378'
def test_novelty_pass_is_immutable_and_authorizes_economics():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()==EXPECTED;x=json.loads(n.OUTPUT.read_text());h=x.pop('manifest_hash');assert n.canonical_hash(x)==h;assert x['source_support_passed'] and x['every_gross9_sleeve_passed'] and x['advance_to_economic_outcomes'];assert not x['evidence_boundary']['outcomes_opened']
