import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_cross_alt_serial_persistence_consensus_relay_economics as e
P=Path('results/high_volatility_cross_alt_serial_persistence_consensus_relay_test_economics_2026-08-13.json');EXPECTED='674b9419b498d021f99cb8b5cc35b91998e160a9f40100c44e01984b67e1a732'
def test_test_failure_is_immutable_terminal_and_sequential():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;r=json.loads(P.read_text());h=r.pop('manifest_hash');assert e.canonical_hash(r)==h;assert r['policy_id']=='HVCASPC-8' and r['stage']=='test' and not r['passed'];assert r['predecessor']['stage']=='train' and r['predecessor']['sha256']=='f987549b3ad67908e0ffa4f5c8ff466b55c637f2bbcc43141452da3b8375eeca';assert r['decision']=='terminal_reject_no_repair' and not r['advance_to_next_stage'] and not r['later_stage_outcomes_opened'];assert r['primary']['base']['absolute_return_pct']<0 and r['primary']['base']['mean_gross_underlying_bp']<20 and r['primary']['cluster_signflip']['pvalue']>.1 and r['primary']['stress']['absolute_return_pct']<0 and r['primary']['calendar_halves']['second']['absolute_return_pct']<0
