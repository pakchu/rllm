import json
from training import build_high_volatility_causal_response_memory_relay_support as support
def test_support_artifact_passes_and_keeps_later_gates_closed():
 p=json.loads(support.RESULT.read_text());assert p['policy_id']=='HVCRMR-12';assert p['support_passed'];assert p['advance_to_gross9_novelty'];assert not p['advance_to_economic_metrics'];assert not p['gross9_rows_opened'];assert [p['support'][n]['events'] for n in support.SPLITS]==[64,244,177,104];h=p.pop('manifest_hash');assert support.chash(p)==h
