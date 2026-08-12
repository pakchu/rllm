import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_cross_alt_path_efficiency_consensus_relay_economics as e
P=Path('results/high_volatility_cross_alt_path_efficiency_consensus_relay_train_economics_2026-08-13.json');EXPECTED='267a1192f51af1b3119135c43d8ba9f13a86671d914c417c08131e68be38e549'
def test_train_failure_is_immutable_terminal_and_sequential():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;r=json.loads(P.read_text());h=r.pop('manifest_hash');assert e.canonical_hash(r)==h;assert r['policy_id']=='HVCAPEC-8' and r['stage']=='train' and not r['passed'];assert r['decision']=='terminal_reject_no_repair' and not r['advance_to_next_stage'] and not r['later_stage_outcomes_opened'];assert r['primary']['base']['absolute_return_pct']>0 and r['primary']['base']['mean_gross_underlying_bp']>20;assert r['primary']['base']['cagr_to_strict_mdd']<3 and r['primary']['cluster_signflip']['pvalue']>.1 and r['primary']['stress']['cagr_to_strict_mdd']<2.5 and r['primary']['calendar_halves']['first']['absolute_return_pct']<0
