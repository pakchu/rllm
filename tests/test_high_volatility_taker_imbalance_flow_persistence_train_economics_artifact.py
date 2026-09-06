import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_taker_imbalance_flow_persistence_economics as economics
RESULT=Path("results/high_volatility_taker_imbalance_flow_persistence_train_economics_2026-08-13.json")
def test_terminal_train_rejection_is_sealed_and_later_stages_absent():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()=="97640bfbc1b981bf9b945cdd769eb4528683a2bba2fb03a45b17713b56d845cd";x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert economics.canonical_hash(x)==h and x['policy_id']=='HVTIFP-8' and x['stage']=='train' and not x['passed'] and x['decision']=='terminal_reject_no_repair' and not x['later_stage_outcomes_opened'];b=x['primary']['base'];assert b['trades']==20 and b['absolute_return_pct']<0 and b['mean_gross_underlying_bp']<20 and b['cagr_to_strict_mdd']<3;assert x['primary']['stress']['absolute_return_pct']<0 and x['primary']['cluster_signflip']['pvalue']>.1 and x['primary']['calendar_halves']['second']['absolute_return_pct']<0
 for stage in ('test','eval','final'):assert not Path(f"results/high_volatility_taker_imbalance_flow_persistence_{stage}_economics_2026-08-13.json").exists()
