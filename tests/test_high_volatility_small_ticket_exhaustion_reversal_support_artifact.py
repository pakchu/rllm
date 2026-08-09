import json
from training import build_high_volatility_small_ticket_exhaustion_reversal_support as support

def test_terminal_support_artifact_is_bound_and_sealed():
 p=json.loads(support.RESULT.read_text());assert p['policy_id']=='HVSTER-8';assert not p['support_passed'];assert not p['advance_to_gross9_novelty'];assert not p['advance_to_economic_outcomes'];assert not p['postentry_return_pnl_execution_price_opened'];assert not p['gross9_rows_opened'];assert [p['support'][n]['events'] for n in support.SPLITS]==[18,2,22,97];h=p.pop('manifest_hash');assert support.prereg.canonical_hash(p)==h
