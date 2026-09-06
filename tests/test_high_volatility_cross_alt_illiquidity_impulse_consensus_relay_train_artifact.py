import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_cross_alt_illiquidity_impulse_consensus_relay_economics as e
P=Path('results/high_volatility_cross_alt_illiquidity_impulse_consensus_relay_train_economics_2026-08-13.json');EXPECTED='02fcdb66b3bfd2a77e8d8cb79706d5f588c6a4b904950a3fc61c50e261862f5c'
def test_train_failure_is_immutable_terminal_and_sequential():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;r=json.loads(P.read_text());h=r.pop('manifest_hash');assert e.canonical_hash(r)==h;assert r['policy_id']=='HVCIIC-8' and r['stage']=='train' and not r['passed'];assert r['decision']=='terminal_reject_no_repair' and not r['advance_to_next_stage'];assert not r['later_stage_outcomes_opened'];assert r['primary']['base']['absolute_return_pct']>0;assert r['primary']['base']['cagr_to_strict_mdd']<3;assert r['primary']['base']['mean_gross_underlying_bp']<20;assert r['primary']['cluster_signflip']['pvalue']>.1;assert r['primary']['stress']['absolute_return_pct']<0
