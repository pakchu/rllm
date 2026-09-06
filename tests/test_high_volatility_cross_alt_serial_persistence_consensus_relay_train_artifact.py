import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_cross_alt_serial_persistence_consensus_relay_economics as e
P=Path('results/high_volatility_cross_alt_serial_persistence_consensus_relay_train_economics_2026-08-13.json');EXPECTED='f987549b3ad67908e0ffa4f5c8ff466b55c637f2bbcc43141452da3b8375eeca'
def test_train_pass_is_immutable_complete_and_sequential():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;r=json.loads(P.read_text());h=r.pop('manifest_hash');assert e.canonical_hash(r)==h;assert r['policy_id']=='HVCASPC-8' and r['stage']=='train' and r['passed'] and r['advance_to_next_stage'];assert not r['later_stage_outcomes_opened'] and all(r['checks'].values());assert r['primary']['base']['absolute_return_pct']>0 and r['primary']['base']['cagr_to_strict_mdd']>=3 and r['primary']['base']['mean_gross_underlying_bp']>=20;assert r['primary']['cluster_signflip']['pvalue']<=.1 and r['primary']['stress']['absolute_return_pct']>0 and r['primary']['stress']['cagr_to_strict_mdd']>=2.5;assert all(x['absolute_return_pct']>0 for x in r['primary']['calendar_halves'].values())
