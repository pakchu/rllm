import json
from training import evaluate_high_volatility_causal_response_memory_relay_gross9_novelty as novelty
def test_terminal_novelty_artifact_is_bound_and_economics_closed():
 p=json.loads(novelty.OUTPUT.read_text());assert p['policy_id']=='HVCRMR-12';assert p['source_support_passed'];assert not p['every_gross9_sleeve_passed'];assert p['gross9_novelty_status']=='failed';assert not p['advance_to_economic_outcomes'];assert p['evidence_boundary']['economic_outcome_rows_opened']==0;assert not p['evidence_boundary']['portfolio_return_or_pnl_metrics_computed'];h=p.pop('manifest_hash');assert novelty.canonical_hash(p)==h
