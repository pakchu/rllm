import hashlib,json
from training import evaluate_high_volatility_leverage_displacement_amplification_reversal_economics as e
EXPECTED='fc7c539da7a6d18474725b080bf72d9213507fac0f350ba3d172bdab21c4fd0c'
def test_train_rejection_is_immutable_and_later_stages_sealed():
 p=e.OUTPUTS['train'];assert hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED;x=json.loads(p.read_text());h=x.pop('manifest_hash');assert e.canonical_hash(x)==h;assert x['stage']=='train' and not x['passed'] and x['decision']=='terminal_reject_no_repair' and not x['later_stage_outcomes_opened'];assert x['primary']['base']['absolute_return_pct']>0 and x['primary']['base']['mean_gross_underlying_bp']>=20 and x['primary']['base']['cagr_to_strict_mdd']<3;assert x['primary']['stress']['absolute_return_pct']>0 and x['primary']['calendar_halves']['first']['absolute_return_pct']<0;assert not e.OUTPUTS['test'].exists() and not e.OUTPUTS['eval'].exists() and not e.OUTPUTS['final'].exists()
