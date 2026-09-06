import json
from training import build_cross_asset_ticket_leadership_relay_support as support
def test_support_artifact_passes_and_later_gates_stay_closed():
 p=json.loads(support.RESULT.read_text());assert p['policy_id']=='CATLR-12';assert p['support_passed'];assert p['advance_to_gross9_novelty'];assert not p['advance_to_economic_outcomes'];assert not p['postentry_return_pnl_execution_price_opened'];assert not p['gross9_rows_opened'];assert [p['support'][n]['events'] for n in support.SPLITS]==[42,75,64,24];h=p.pop('manifest_hash');assert support.chash(p)==h
