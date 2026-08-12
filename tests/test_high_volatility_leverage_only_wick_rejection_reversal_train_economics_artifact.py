import hashlib,json
from training import evaluate_high_volatility_leverage_only_wick_rejection_reversal_economics as e
EXPECTED='2a03d0c27b1823c3ca3f3fe4f6cd7da281609a7660f690ce79038c096bb4a525'
def test_train_rejection_is_immutable_and_later_stages_sealed():
 p=e.OUTPUTS['train'];assert hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED;x=json.loads(p.read_text());h=x.pop('manifest_hash');assert e.canonical_hash(x)==h;assert x['stage']=='train' and not x['passed'] and x['decision']=='terminal_reject_no_repair' and not x['later_stage_outcomes_opened'];assert not x['advance_to_next_stage'] and not x['advance_to_post_stage_volatility_audit'];assert x['primary']['base']['absolute_return_pct']>0 and x['primary']['base']['mean_gross_underlying_bp']>=20 and x['primary']['base']['cagr_to_strict_mdd']<3;assert x['primary']['calendar_halves']['first']['absolute_return_pct']<0;assert not e.OUTPUTS['test'].exists() and not e.OUTPUTS['eval'].exists() and not e.OUTPUTS['final'].exists()
