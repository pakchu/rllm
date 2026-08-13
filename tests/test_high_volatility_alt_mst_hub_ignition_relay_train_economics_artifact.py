import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_alt_mst_hub_ignition_relay_economics as e
RESULT=Path('results/high_volatility_alt_mst_hub_ignition_relay_train_economics_2026-08-13.json');EXPECTED='3d2d4a5842c6931cc962ce7832679f3bb0333af62bcf65e38e570cbef0c1aff0'
def test_terminal_train_rejection_is_immutable_and_later_stages_absent():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED
 x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert e.canonical_hash(x)==h
 assert x['policy_id']=='HVAMST-8' and x['stage']=='train' and not x['passed'] and x['decision']=='terminal_reject_no_repair' and not x['later_stage_outcomes_opened']
 b=x['primary']['base'];assert b['trades']==18 and b['absolute_return_pct']<0 and b['mean_gross_underlying_bp']<20 and b['cagr_to_strict_mdd']<3
 assert x['primary']['stress']['absolute_return_pct']<0 and x['primary']['cluster_signflip']['pvalue']>.1
 assert x['primary']['calendar_halves']['first']['absolute_return_pct']>0 and x['primary']['calendar_halves']['second']['absolute_return_pct']<0
 for stage in ('test','eval','final'):assert not Path(f'results/high_volatility_alt_mst_hub_ignition_relay_{stage}_economics_2026-08-13.json').exists()
