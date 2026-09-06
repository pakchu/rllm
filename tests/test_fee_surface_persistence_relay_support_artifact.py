import json
from training import build_fee_surface_persistence_relay_support as support
def test_terminal_support_artifact_is_bound_and_sealed():
 p=json.loads(support.RESULT.read_text());assert p['policy_id']=='FSPR-24';assert not p['support_passed'];assert not p['advance_to_gross9_novelty'];assert not p['advance_to_economic_outcomes'];assert not p['postentry_return_pnl_execution_price_opened'];assert not p['gross9_rows_opened'];assert [p['support'][n]['events'] for n in support.SPLITS]==[5,10,3,1];h=p.pop('manifest_hash');assert support.chash(p)==h
